"""
Remap roof metadata (normals + area) from custom roof polygons onto existing radiation sensors.

This utility is designed for experiments where you want to keep radiation outputs unchanged
(`*_insolation_Whm2.feather`) but adjust roof orientation and area used by PV calculations.
It preserves SURFACE ids and row counts in `*_geometry.csv`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

import cea.inputlocator


@dataclass
class RoofFace:
    building: str
    roof_id: str
    polygon_xy: Polygon
    normal: np.ndarray
    area_3d: float


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def polygon_normal(coords_xyz: np.ndarray) -> np.ndarray:
    a = coords_xyz[0]
    b = coords_xyz[1]
    c = coords_xyz[2]
    n = unit(np.cross(b - a, c - a))
    # Keep upward convention for compatibility with CEA roof orientation assumptions.
    if n[2] < 0:
        n = -n
    return n


def polygon_area_3d(coords_xyz: np.ndarray) -> float:
    """Compute 3D area of a planar polygon by fan triangulation."""
    origin = coords_xyz[0]
    area = 0.0
    for i in range(1, len(coords_xyz) - 2):
        u = coords_xyz[i] - origin
        v = coords_xyz[i + 1] - origin
        area += np.linalg.norm(np.cross(u, v)) * 0.5
    return float(area)


def iter_polygons(geom):
    if geom is None:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield poly


def load_custom_roof_faces(roof_file: str, target_crs) -> dict[str, list[RoofFace]]:
    roof_gdf = gpd.read_file(roof_file)
    if roof_gdf.empty:
        return {}
    if "building" not in roof_gdf.columns:
        raise ValueError("Custom roof file must include a 'building' column.")

    if roof_gdf.crs is not None and target_crs is not None and roof_gdf.crs != target_crs:
        roof_gdf = roof_gdf.to_crs(target_crs)

    result: dict[str, list[RoofFace]] = {}
    for _, row in roof_gdf.iterrows():
        bname = str(row["building"])
        roof_id = str(row["roof_id"]) if "roof_id" in roof_gdf.columns else "roof"
        faces = result.setdefault(bname, [])

        for poly in iter_polygons(row.geometry):
            coords = np.array(poly.exterior.coords, dtype=float)
            if coords.shape[0] < 4 or coords.shape[1] < 3:
                continue

            coords_xyz = coords[:, :3]
            normal = polygon_normal(coords_xyz)
            area = polygon_area_3d(coords_xyz)
            if area <= 0:
                continue

            xy = [(float(x), float(y)) for x, y, *_ in coords]
            polygon_xy = Polygon(xy)
            if not polygon_xy.is_valid or polygon_xy.area == 0:
                continue

            faces.append(
                RoofFace(
                    building=bname,
                    roof_id=roof_id,
                    polygon_xy=polygon_xy,
                    normal=normal,
                    area_3d=area,
                )
            )
    return result


def affinity(face: RoofFace, point_xy: Point) -> float:
    # Distance to face footprint in XY. 0 means sensor is projected inside face.
    d = float(face.polygon_xy.distance(point_xy))
    return 1.0 / (d + 1e-6)


def remap_building_metadata(df: pd.DataFrame, faces: list[RoofFace]) -> pd.DataFrame:
    roof_mask = df["TYPE"] == "roofs"
    roof_df = df.loc[roof_mask].copy()
    if roof_df.empty:
        return df
    if not faces:
        return df

    sensor_xy = roof_df[["Xcoor", "Ycoor"]].to_numpy(dtype=float)
    n_sensors = sensor_xy.shape[0]
    n_faces = len(faces)

    # A[face, sensor] = affinity
    A = np.zeros((n_faces, n_sensors), dtype=float)
    for i, face in enumerate(faces):
        for j in range(n_sensors):
            p = Point(sensor_xy[j, 0], sensor_xy[j, 1])
            A[i, j] = affinity(face, p)

    # For each face, distribute its area over existing sensors.
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = A / row_sums  # P[face, sensor]

    face_areas = np.array([f.area_3d for f in faces], dtype=float)
    normals = np.array([f.normal for f in faces], dtype=float)  # [face, xyz]

    # Area_j = sum_i P[i,j] * area_i (exact area conservation)
    area_per_sensor = (P.T @ face_areas)  # [sensor]

    # Normal_j = unit(sum_i P[i,j] * area_i * n_i)
    weighted = (P.T * face_areas) @ normals  # [sensor, xyz]
    normals_per_sensor = np.array([unit(v) for v in weighted], dtype=float)

    roof_df.loc[:, "AREA_m2"] = area_per_sensor
    roof_df.loc[:, "Xdir"] = normals_per_sensor[:, 0]
    roof_df.loc[:, "Ydir"] = normals_per_sensor[:, 1]
    roof_df.loc[:, "Zdir"] = normals_per_sensor[:, 2]
    roof_df.loc[:, "orientation"] = "top"

    # Recombine
    out = df.copy()
    out.loc[roof_mask, ["AREA_m2", "Xdir", "Ydir", "Zdir", "orientation"]] = roof_df[
        ["AREA_m2", "Xdir", "Ydir", "Zdir", "orientation"]
    ].values

    return out


def output_path(metadata_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(metadata_path)
    return f"{base}{suffix}{ext}"


def parse_args():
    parser = argparse.ArgumentParser(description="Remap roof metadata for PV experiments.")
    parser.add_argument("--scenario", required=True, help="Scenario path")
    parser.add_argument(
        "--roof-file",
        default="",
        help="Optional custom roof file path. Default: <scenario>/inputs/building-geometry/roof_surfaces.geojson",
    )
    parser.add_argument(
        "--building",
        default="",
        help="Optional single building name. Default: process all buildings found in roof file.",
    )
    parser.add_argument(
        "--suffix",
        default=".remap",
        help="Output suffix for metadata files when not using --in-place (default: .remap)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original metadata file and write a .bak backup first.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    locator = cea.inputlocator.InputLocator(args.scenario)

    zone_path = locator.get_zone_geometry()
    zone_gdf = gpd.read_file(zone_path)
    target_crs = zone_gdf.crs

    roof_file = args.roof_file or os.path.join(
        args.scenario, "inputs", "building-geometry", "roof_surfaces.geojson"
    )
    if not os.path.exists(roof_file):
        raise FileNotFoundError(f"Custom roof file not found: {roof_file}")

    roofs_by_building = load_custom_roof_faces(roof_file, target_crs)
    if not roofs_by_building:
        raise ValueError("No valid custom roof faces were loaded.")

    buildings = [args.building] if args.building else sorted(roofs_by_building.keys())
    for building in buildings:
        if building not in roofs_by_building:
            print(f"[skip] {building}: no custom roof faces loaded.")
            continue

        metadata_path = locator.get_radiation_metadata(building)
        if not os.path.exists(metadata_path):
            print(f"[skip] {building}: metadata not found at {metadata_path}")
            continue

        df = pd.read_csv(metadata_path)
        if "SURFACE" not in df.columns:
            print(f"[skip] {building}: invalid metadata format (missing SURFACE).")
            continue

        before_roof = df.loc[df["TYPE"] == "roofs", "AREA_m2"].sum()
        remapped = remap_building_metadata(df, roofs_by_building[building])
        after_roof = remapped.loc[remapped["TYPE"] == "roofs", "AREA_m2"].sum()
        custom_roof_total = sum(face.area_3d for face in roofs_by_building[building])

        if args.in_place:
            backup = metadata_path + ".bak"
            if not os.path.exists(backup):
                shutil.copy2(metadata_path, backup)
            remapped.to_csv(metadata_path, index=False)
            out = metadata_path
        else:
            out = output_path(metadata_path, args.suffix)
            remapped.to_csv(out, index=False)

        print(
            f"[ok] {building}: {out}\n"
            f"     roof area before={before_roof:.3f} m2, after={after_roof:.3f} m2, custom_total={custom_roof_total:.3f} m2"
        )


if __name__ == "__main__":
    main()
