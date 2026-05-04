"""Build validation datasets from Lisbon radiation tiles and CEA zone files.

Two-step workflow:

1) create-areas:
   - reads GeoTIFF extents
   - converts bounds to WGS84 (EPSG:4326)
   - writes one polygon GeoJSON per area
   - writes a JSON/YAML config with area metadata

2) build-validation:
    - loads areas config
    - filters CEA zone.shp buildings to each area
    - keeps only buildings fully inside raster extent
    - classifies residential buildings from CEA usage fields
    - sums raster radiation within each kept residential building footprint
      and reports annual kWh
    - writes per-area validation outputs
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.mask import mask
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform

try:
    import yaml
except ImportError:  # pragma: no cover - optional fallback
    yaml = None


DEFAULT_CEA_NAME_FIELD = "name"
DEFAULT_NODATA_FALLBACK = -9999.0
DEFAULT_RASTER_EDGE_BUFFER_M = 12
WH_TO_KWH = 1.0 / 1000.0
WEB_MERCATOR_EPSG = 3857

CEA_RESIDENTIAL_HINTS = {
    "res",
    "residential",
    "single_res",
    "multi_res",
    "apartment",
    "apartments",
    "house",
}

CEA_USAGE_FIELDS = ("use_type1", "use_type2", "use_type3", "resi_type")


@dataclass
class AreaConfig:
    area_id: str
    raster_path: Path
    polygon_geojson_path: Path
    zone_shp_path: Path | None
    cea_name_field: str = DEFAULT_CEA_NAME_FIELD


def _normalize_area_id(stem: str) -> str:
    stem = re.sub(r"^global_anual_", "", stem, flags=re.IGNORECASE)
    stem = stem.strip("_")
    stem = re.sub(r"\s+", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem or "area"


def _to_absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _load_config(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files, but it is not installed.")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if not isinstance(payload, list):
        raise ValueError("Areas config must be a list of area objects.")
    return payload


def _dump_config(path: Path, payload: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML config files, but it is not installed.")
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def _feature_collection_for_polygon(
    area_id: str,
    polygon: Polygon,
    raster_path: Path,
    source_crs: str,
) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "name": f"{area_id}_polygon",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "area_id": area_id,
                    "raster_path": str(_to_absolute(raster_path)),
                    "source_crs": source_crs,
                    "crs": "EPSG:4326",
                    "ucea_compatible": True,
                },
                "geometry": mapping(polygon),
            }
        ],
    }


def _write_geojson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_polygon_geojson(path: Path) -> Polygon:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not features:
        raise ValueError(f"No features found in polygon GeoJSON: {path}")
    geom = shape(features[0]["geometry"])
    if geom.geom_type != "Polygon":
        raise ValueError(f"Expected Polygon geometry in {path}, got {geom.geom_type}.")
    return geom


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_residential_text(value: Any, hints: set[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    # Match hints as full terms/tokens to avoid false positives like
    # "restaurant" matching the hint "res".
    for hint in hints:
        pattern = rf"\b{re.escape(hint.lower())}\b"
        if re.search(pattern, text):
            return True
    return False


def _is_residential_from_cea(row: pd.Series) -> bool:
    for col in CEA_USAGE_FIELDS:
        if _looks_residential_text(row.get(col), CEA_RESIDENTIAL_HINTS):
            return True
    return False


def _classify_residential(row: pd.Series) -> tuple[bool, str]:
    if _is_residential_from_cea(row):
        return True, "cea"
    return False, ""


def _ensure_geometry_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    allowed = {"Polygon", "MultiPolygon"}
    mask_geom = gdf.geometry.geom_type.isin(allowed) & (~gdf.geometry.is_empty) & gdf.geometry.notna()
    return gdf.loc[mask_geom].copy()


def _as_pyproj_crs(crs_like: Any) -> CRS | None:
    if crs_like is None:
        return None
    try:
        return CRS.from_user_input(crs_like)
    except Exception:
        return None


def _is_metric_projected_crs(crs_like: Any) -> bool:
    crs = _as_pyproj_crs(crs_like)
    if crs is None or not crs.is_projected:
        return False

    axis_info = list(getattr(crs, "axis_info", []) or [])
    if not axis_info:
        return False

    unit_names = [str(getattr(axis, "unit_name", "")).strip().lower() for axis in axis_info[:2]]
    return all(("metre" in unit_name) or ("meter" in unit_name) for unit_name in unit_names)


def _is_web_mercator(crs_like: Any) -> bool:
    crs = _as_pyproj_crs(crs_like)
    if crs is None:
        return False
    epsg = crs.to_epsg()
    return epsg == WEB_MERCATOR_EPSG


def _utm_epsg_from_lon_lat(lon: float, lat: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def _select_area_crs(zone_crs_like: Any, raster_crs_like: Any, area_polygon_wgs84: Polygon) -> CRS:
    zone_crs = _as_pyproj_crs(zone_crs_like)
    if zone_crs is not None and _is_metric_projected_crs(zone_crs) and not _is_web_mercator(zone_crs):
        return zone_crs

    raster_crs = _as_pyproj_crs(raster_crs_like)
    if raster_crs is not None and _is_metric_projected_crs(raster_crs) and not _is_web_mercator(raster_crs):
        return raster_crs

    centroid = area_polygon_wgs84.centroid
    utm_epsg = _utm_epsg_from_lon_lat(float(centroid.x), float(centroid.y))
    return CRS.from_epsg(utm_epsg)


def _load_area_configs(config_path: Path) -> list[AreaConfig]:
    rows = _load_config(config_path)
    areas: list[AreaConfig] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Config entry at index {idx} must be an object.")

        area_id = str(row.get("area_id") or "").strip()
        raster_path = str(row.get("raster_path") or "").strip()
        polygon_path = str(row.get("polygon_geojson_path") or "").strip()
        zone_path = str(row.get("zone_shp_path") or "").strip()
        cea_name_field = str(row.get("cea_name_field") or DEFAULT_CEA_NAME_FIELD).strip()

        if not area_id:
            raise ValueError(f"Config entry at index {idx} is missing 'area_id'.")
        if not raster_path:
            raise ValueError(f"Config entry '{area_id}' is missing 'raster_path'.")
        if not polygon_path:
            raise ValueError(f"Config entry '{area_id}' is missing 'polygon_geojson_path'.")

        areas.append(
            AreaConfig(
                area_id=area_id,
                raster_path=_to_absolute(Path(raster_path)),
                polygon_geojson_path=_to_absolute(Path(polygon_path)),
                zone_shp_path=_to_absolute(Path(zone_path)) if zone_path else None,
                cea_name_field=cea_name_field or DEFAULT_CEA_NAME_FIELD,
            )
        )
    return areas


def _prepare_zone_buildings(
    zone_gdf: gpd.GeoDataFrame,
    cea_name_field: str,
    intersection_areas_m2: pd.Series | None = None,
) -> gpd.GeoDataFrame:
    zone_gdf = zone_gdf.copy()
    zone_gdf["zone_uid"] = zone_gdf.index.astype(str)
    if cea_name_field not in zone_gdf.columns:
        fallback = next((c for c in ("name", "Name", "NAME") if c in zone_gdf.columns), None)
        if fallback is None:
            zone_gdf[cea_name_field] = zone_gdf["zone_uid"]
        else:
            cea_name_field = fallback

    keep_zone_cols = ["zone_uid", cea_name_field, "geometry"]
    for col in CEA_USAGE_FIELDS:
        if col in zone_gdf.columns:
            keep_zone_cols.append(col)
    zone_subset = zone_gdf[keep_zone_cols].copy()
    if zone_subset.empty:
        merged = gpd.GeoDataFrame(
            columns=["osm_uid", "osmid", "element_type", "building", "cea_name", "intersection_area_m2", "geometry"],
            geometry="geometry",
            crs=zone_gdf.crs,
        )
        for col in CEA_USAGE_FIELDS:
            if col not in merged.columns:
                merged[col] = None
        return merged

    zone_subset = zone_subset.rename(columns={cea_name_field: "cea_name"})
    zone_subset["osm_uid"] = "cea/" + zone_subset["zone_uid"]
    zone_subset["osmid"] = zone_subset["zone_uid"]
    zone_subset["element_type"] = "cea_zone"
    zone_subset["building"] = "yes"
    if intersection_areas_m2 is not None:
        zone_subset["intersection_area_m2"] = pd.to_numeric(
            intersection_areas_m2.reindex(zone_subset.index), errors="coerce"
        ).fillna(0.0)
    else:
        zone_subset["intersection_area_m2"] = zone_subset.geometry.area

    keep_cols = ["osm_uid", "osmid", "element_type", "building", "cea_name", "intersection_area_m2"]
    keep_cols += [c for c in CEA_USAGE_FIELDS if c in zone_subset.columns]
    keep_cols.append("geometry")
    return gpd.GeoDataFrame(zone_subset[keep_cols], geometry="geometry", crs=zone_subset.crs)


def _sum_radiation_for_geometry(dataset: rasterio.DatasetReader, geometry) -> tuple[float, int]:
    nodata = dataset.nodata if dataset.nodata is not None else DEFAULT_NODATA_FALLBACK
    out, _ = mask(dataset, [geometry], crop=True, filled=False)
    band = out[0]

    if np.ma.isMaskedArray(band):
        values = np.asarray(band.compressed(), dtype=float)
    else:
        values = np.asarray(band, dtype=float).ravel()

    if values.size == 0:
        return 0.0, 0

    valid = np.isfinite(values) & (values != float(nodata))
    valid_values = values[valid]
    if valid_values.size == 0:
        return 0.0, 0
    return float(valid_values.sum()), int(valid_values.size)


def _geometry_fully_inside_raster_extent(
    dataset: rasterio.DatasetReader,
    geometry,
    raster_edge_buffer_m: float = 0.0,
) -> bool:
    bounds = dataset.bounds
    raster_extent_polygon = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    if raster_edge_buffer_m > 0.0:
        raster_extent_polygon = raster_extent_polygon.buffer(-float(raster_edge_buffer_m))
        if raster_extent_polygon.is_empty:
            return False
    return geometry.covered_by(raster_extent_polygon)


def _write_gdf_geojson(path: Path, gdf: gpd.GeoDataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gdf.to_json(drop_id=True), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_areas(tiles_dir: Path, config_out: Path, polygons_dir: Path | None = None) -> None:
    tiles_dir = _to_absolute(tiles_dir)
    config_out = _to_absolute(config_out)
    polygons_dir = _to_absolute(polygons_dir) if polygons_dir else config_out.parent
    polygons_dir.mkdir(parents=True, exist_ok=True)

    tif_paths = sorted(tiles_dir.glob("*.tif"))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {tiles_dir}")

    config_payload: list[dict[str, Any]] = []
    for tif_path in tif_paths:
        area_id = _normalize_area_id(tif_path.stem)

        with rasterio.open(tif_path) as ds:
            if ds.crs is None:
                raise ValueError(f"Raster has no CRS: {tif_path}")
            bounds = ds.bounds
            source_crs = ds.crs.to_string()
            bounds_polygon = Polygon(
                [
                    (bounds.left, bounds.top),
                    (bounds.right, bounds.top),
                    (bounds.right, bounds.bottom),
                    (bounds.left, bounds.bottom),
                    (bounds.left, bounds.top),
                ]
            )
            transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
            polygon_wgs84 = transform(transformer.transform, bounds_polygon)

        polygon_path = polygons_dir / f"{area_id}_polygon.geojson"
        geojson_payload = _feature_collection_for_polygon(area_id, polygon_wgs84, tif_path, source_crs)
        _write_geojson(polygon_path, geojson_payload)

        config_payload.append(
            {
                "area_id": area_id,
                "raster_path": str(_to_absolute(tif_path)),
                "polygon_geojson_path": str(_to_absolute(polygon_path)),
                "zone_shp_path": "",
                "cea_name_field": DEFAULT_CEA_NAME_FIELD,
            }
        )

    _dump_config(config_out, config_payload)
    print(f"Wrote {len(config_payload)} area polygons to {polygons_dir}")
    print(f"Wrote config template to {config_out}")
    print("Next step: fill each 'zone_shp_path' and run 'build-validation'.")


def build_validation(
    areas_config: Path,
    output_dir: Path,
    raster_edge_buffer_m: float = DEFAULT_RASTER_EDGE_BUFFER_M,
) -> None:
    areas = _load_area_configs(_to_absolute(areas_config))
    output_dir = _to_absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raster_edge_buffer_m = float(raster_edge_buffer_m)
    if raster_edge_buffer_m < 0.0:
        raise ValueError("--raster-edge-buffer-m must be >= 0.")

    for area in areas:
        print(f"\nProcessing area: {area.area_id}")
        if area.zone_shp_path is None:
            raise ValueError(
                f"Area '{area.area_id}' has empty zone_shp_path. Fill it in {areas_config} before running build-validation."
            )

        if not area.raster_path.exists():
            raise FileNotFoundError(f"Raster not found for area '{area.area_id}': {area.raster_path}")
        if not area.polygon_geojson_path.exists():
            raise FileNotFoundError(f"Polygon GeoJSON not found for area '{area.area_id}': {area.polygon_geojson_path}")
        if not area.zone_shp_path.exists():
            raise FileNotFoundError(f"zone.shp not found for area '{area.area_id}': {area.zone_shp_path}")

        area_polygon_wgs84 = _read_polygon_geojson(area.polygon_geojson_path)

        with rasterio.open(area.raster_path) as raster_ds:
            raster_crs = raster_ds.crs
            if raster_crs is None:
                raise ValueError(f"Raster has no CRS: {area.raster_path}")
            _write_geojson(
                output_dir / f"{area.area_id}_polygon.geojson",
                _feature_collection_for_polygon(
                    area.area_id, area_polygon_wgs84, area.raster_path, raster_crs.to_string()
                ),
            )

            zone_gdf = gpd.read_file(area.zone_shp_path)
            if zone_gdf.empty:
                raise ValueError(f"zone.shp is empty for area '{area.area_id}': {area.zone_shp_path}")
            zone_gdf = _ensure_geometry_types(zone_gdf)
            area_crs = _select_area_crs(zone_gdf.crs, raster_crs, area_polygon_wgs84)

            zone_gdf_raster = zone_gdf.to_crs(raster_crs)
            zone_gdf_area = zone_gdf.to_crs(area_crs)

            area_poly_raster = gpd.GeoSeries([area_polygon_wgs84], crs="EPSG:4326").to_crs(raster_crs).iloc[0]
            # Keep only buildings whose centroid lies inside the area polygon.
            # This is the intended validation-area membership rule.
            centroid_inside_mask = zone_gdf_raster.geometry.centroid.within(area_poly_raster)
            zone_in_area = zone_gdf_raster.loc[centroid_inside_mask].copy()
            _write_gdf_geojson(
                output_dir / f"{area.area_id}_zone_buildings_raw.geojson",
                gpd.GeoDataFrame(zone_in_area, geometry="geometry", crs=raster_crs).to_crs("EPSG:4326"),
            )

            fully_inside_raster_mask = zone_in_area.geometry.apply(
                lambda geom: _geometry_fully_inside_raster_extent(
                    raster_ds, geom, raster_edge_buffer_m=raster_edge_buffer_m
                )
            )
            zone_outside_raster_extent = zone_in_area.loc[~fully_inside_raster_mask].copy()
            zone_in_area = zone_in_area.loc[fully_inside_raster_mask].copy()

            if zone_in_area.empty:
                empty_geo = gpd.GeoDataFrame(
                    columns=[
                        "area_id",
                        "osm_uid",
                        "osmid",
                        "element_type",
                        "building",
                        "cea_name",
                        "intersection_area_m2",
                        "is_residential",
                        "residential_source",
                        "radiation_sum_kwh_year",
                        "radiation_sum",
                        "radiation_point_count",
                        "geometry",
                    ],
                    geometry="geometry",
                    crs="EPSG:4326",
                )
                _write_gdf_geojson(output_dir / f"{area.area_id}_buildings_with_cea.geojson", empty_geo)
                _write_gdf_geojson(output_dir / f"{area.area_id}_residential_buildings.geojson", empty_geo)
                empty_geo.drop(columns=["geometry"]).to_csv(
                    output_dir / f"{area.area_id}_residential_radiation.csv", index=False
                )
                print("  No CEA zone buildings found after centroid-in-area and full-raster-extent filters.")
                continue

            zone_in_area_area_crs = zone_gdf_area.loc[zone_in_area.index].copy()
            # Respect zone.shp as the reference geometry area for each selected building.
            intersection_areas_m2 = zone_in_area_area_crs.geometry.area
            buildings = _prepare_zone_buildings(
                zone_in_area,
                area.cea_name_field,
                intersection_areas_m2=intersection_areas_m2,
            )
            buildings["area_id"] = area.area_id

            classifications = buildings.apply(_classify_residential, axis=1, result_type="expand")
            buildings["is_residential"] = classifications[0]
            buildings["residential_source"] = classifications[1]

            buildings["radiation_sum_kwh_year"] = np.nan
            buildings["radiation_sum"] = np.nan
            buildings["radiation_point_count"] = np.nan

            residential_idx = buildings.index[buildings["is_residential"]]
            for idx in residential_idx:
                geom = buildings.at[idx, "geometry"]
                total, count = _sum_radiation_for_geometry(raster_ds, geom)
                radiation_kwh = _safe_float(total * WH_TO_KWH)
                buildings.at[idx, "radiation_sum_kwh_year"] = radiation_kwh
                # Backward-compatible alias kept for existing consumers.
                buildings.at[idx, "radiation_sum"] = radiation_kwh
                buildings.at[idx, "radiation_point_count"] = int(count)

            keep_cols = [
                "area_id",
                "osm_uid",
                "osmid",
                "element_type",
                "building",
                "cea_name",
                "intersection_area_m2",
                "is_residential",
                "residential_source",
                "radiation_sum_kwh_year",
                "radiation_sum",
                "radiation_point_count",
            ]
            for col in CEA_USAGE_FIELDS:
                if col in buildings.columns:
                    keep_cols.append(col)
            keep_cols.append("geometry")

            buildings_out = gpd.GeoDataFrame(buildings[keep_cols], geometry="geometry", crs=raster_crs).to_crs("EPSG:4326")
            residential_all_out = buildings_out[buildings_out["is_residential"]].copy()
            zero_radiation_mask = (
                pd.to_numeric(residential_all_out["radiation_sum_kwh_year"], errors="coerce").fillna(0.0) <= 0.0
            )
            removed_zero_radiation = residential_all_out.loc[zero_radiation_mask].copy()
            residential_out = residential_all_out.loc[~zero_radiation_mask].copy()

            _write_gdf_geojson(output_dir / f"{area.area_id}_buildings_with_cea.geojson", buildings_out)
            _write_gdf_geojson(output_dir / f"{area.area_id}_residential_buildings.geojson", residential_out)

            csv_cols = [c for c in residential_out.columns if c != "geometry"]
            residential_out[csv_cols].to_csv(output_dir / f"{area.area_id}_residential_radiation.csv", index=False)

            removed_buildings = sorted(
                {
                    str(value).strip()
                    for value in removed_zero_radiation["cea_name"].tolist()
                    if str(value).strip()
                }
            )
            summary_payload = {
                "area_id": area.area_id,
                "zone_buildings_total": int(len(buildings_out)),
                "raster_edge_buffer_m": raster_edge_buffer_m,
                "removed_outside_raster_extent_buildings_count": int(len(zone_outside_raster_extent)),
                "residential_before_zero_radiation_filter": int(len(residential_all_out)),
                "removed_zero_radiation_buildings_count": int(len(removed_zero_radiation)),
                "removed_zero_radiation_buildings": removed_buildings,
                "residential_kept_after_zero_radiation_filter": int(len(residential_out)),
            }
            _write_json(output_dir / f"{area.area_id}_validation_summary.json", summary_payload)

            print(f"  CEA zone buildings inside area: {len(buildings_out)}")
            print(
                "  Removed (outside buffered raster extent, "
                f"buffer={raster_edge_buffer_m:.2f} m): {len(zone_outside_raster_extent)}"
            )
            print(f"  Residential buildings before zero-radiation filter: {len(residential_all_out)}")
            print(f"  Removed due to zero radiation: {len(removed_zero_radiation)}")
            print(f"  Residential buildings kept: {len(residential_out)}")
            print(f"  Area summary: {output_dir / f'{area.area_id}_validation_summary.json'}")

    print(f"\nValidation outputs written to: {output_dir}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_create = subparsers.add_parser(
        "create-areas",
        help="Generate per-tile area polygons and an areas config template.",
    )
    p_create.add_argument("--tiles-dir", type=Path, required=True, help="Directory containing input GeoTIFF tiles.")
    p_create.add_argument(
        "--config-out",
        type=Path,
        required=True,
        help="Output areas config path (.json, .yaml, or .yml).",
    )
    p_create.add_argument(
        "--polygons-dir",
        type=Path,
        default=None,
        help="Optional output directory for area polygon GeoJSON files (defaults to config directory).",
    )

    p_build = subparsers.add_parser(
        "build-validation",
        help="Build per-area validation dataset using CEA zone.shp files.",
    )
    p_build.add_argument("--areas-config", type=Path, required=True, help="Areas config produced by create-areas.")
    p_build.add_argument("--output-dir", type=Path, required=True, help="Directory for per-area output files.")
    p_build.add_argument(
        "--raster-edge-buffer-m",
        type=float,
        default=DEFAULT_RASTER_EDGE_BUFFER_M,
        help=(
            "Inset buffer in meters (raster CRS units) applied to raster bounds before "
            "keeping buildings. Helps exclude buildings near black/no-data frames."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "create-areas":
        create_areas(args.tiles_dir, args.config_out, args.polygons_dir)
        return

    if args.command == "build-validation":
        build_validation(
            args.areas_config,
            args.output_dir,
            raster_edge_buffer_m=args.raster_edge_buffer_m,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
