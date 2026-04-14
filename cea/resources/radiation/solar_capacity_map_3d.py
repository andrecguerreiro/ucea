"""
Create an interactive 3D map of all zone buildings with solar-capacity bars.

This script is intentionally standalone (run after CEA workflows finish) and
does not require registering as a CEA CLI script.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np

from cea.inputlocator import InputLocator
from cea.resources.radiation.workflow_metrics_report import normalise_panel_ids, to_float


DEFAULT_MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DEFAULT_BAR_SCALE_M_PER_MWH = 0.02
DEFAULT_MIN_BAR_HEIGHT_M = 1.5
DEFAULT_BAR_WIDTH_M = 9.0
DEFAULT_BUILDING_OPACITY = 180
DEFAULT_BAR_OPACITY = 230
DEFAULT_SHOW_CUSTOM_ROOFS = True
DEFAULT_CUSTOM_ROOF_Z_OFFSET_M = 0.25


def _ordered_ring_from_points_3d(points_3d: list[list[float]]) -> list[list[float]] | None:
    if len(points_3d) < 3:
        return None

    cleaned: list[list[float]] = []
    for point in points_3d:
        if not isinstance(point, list) or len(point) < 3:
            continue
        cleaned.append([float(point[0]), float(point[1]), float(point[2])])

    if len(cleaned) < 3:
        return None

    # Drop duplicated closure before ordering; closure is re-added at the end.
    if cleaned[0] == cleaned[-1]:
        cleaned = cleaned[:-1]
    if len(cleaned) < 3:
        return None

    pts = np.unique(np.round(np.asarray(cleaned, dtype=float), 9), axis=0)
    if pts.shape[0] < 3:
        return None

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis_u = vh[0]
    normal = vh[2]
    axis_v = np.cross(normal, axis_u)

    norm_u = np.linalg.norm(axis_u)
    norm_v = np.linalg.norm(axis_v)
    if norm_u == 0.0 or norm_v == 0.0:
        return None

    axis_u = axis_u / norm_u
    axis_v = axis_v / norm_v

    uv = np.column_stack(
        [
            np.dot(centered, axis_u),
            np.dot(centered, axis_v),
        ]
    )
    angles = np.arctan2(uv[:, 1], uv[:, 0])
    order = np.argsort(angles)
    ordered = pts[order]

    if ordered.shape[0] < 3:
        return None

    ring = ordered.tolist()
    ring.append(ordered[0].tolist())
    return [[float(p[0]), float(p[1]), float(p[2])] for p in ring]


def _import_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'geopandas'. Install CEA dependencies and rerun."
        ) from exc
    return gpd


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pandas'. Install CEA dependencies and rerun."
        ) from exc
    return pd


def _import_pyproj_transformer():
    try:
        from pyproj import Transformer  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'pyproj'. Install CEA dependencies and rerun."
        ) from exc
    return Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an interactive 3D map visualiser for a scenario with all buildings and "
            "solar-capacity bars."
        )
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to CEA scenario.",
    )
    parser.add_argument(
        "--pv-panel",
        default="PV1",
        help="PV panel id used to read PV total buildings CSV (default: PV1).",
    )
    parser.add_argument(
        "--output-html",
        default="",
        help=(
            "Output HTML path. Default: "
            "<scenario>/outputs/data/solar-radiation/solar_capacity_3d_map.html"
        ),
    )
    parser.add_argument(
        "--bar-scale-m-per-mwh",
        type=float,
        default=DEFAULT_BAR_SCALE_M_PER_MWH,
        help=(
            "Initial bar scale in metres per MWh/year of PV generation "
            f"(default: {DEFAULT_BAR_SCALE_M_PER_MWH})."
        ),
    )
    parser.add_argument(
        "--min-bar-height-m",
        type=float,
        default=DEFAULT_MIN_BAR_HEIGHT_M,
        help=f"Minimum visible bar height in metres for non-zero capacity (default: {DEFAULT_MIN_BAR_HEIGHT_M}).",
    )
    parser.add_argument(
        "--bar-width-m",
        type=float,
        default=DEFAULT_BAR_WIDTH_M,
        help=f"Bar width in metres (default: {DEFAULT_BAR_WIDTH_M}).",
    )
    parser.add_argument(
        "--buildings",
        default="",
        help="Optional comma-separated building filter (e.g. B1000,B1001).",
    )
    parser.add_argument(
        "--roof-file",
        default="",
        help=(
            "Optional custom roof GeoJSON path. Default: "
            "<scenario>/inputs/building-geometry/roof_surfaces.geojson"
        ),
    )
    parser.add_argument(
        "--show-custom-roofs",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_CUSTOM_ROOFS,
        help="Show custom roof planes when a valid roof GeoJSON exists (use --no-show-custom-roofs to disable).",
    )
    parser.add_argument(
        "--custom-roof-z-offset-m",
        type=float,
        default=DEFAULT_CUSTOM_ROOF_Z_OFFSET_M,
        help=(
            "Minimum vertical offset above building top used to keep custom roof surfaces visible "
            f"(default: {DEFAULT_CUSTOM_ROOF_Z_OFFSET_M})."
        ),
    )
    parser.add_argument(
        "--map-style",
        default=DEFAULT_MAP_STYLE,
        help="Map style URL for deck.gl.",
    )
    return parser.parse_args()


def parse_building_filter(buildings_arg: str) -> set[str] | None:
    names = [token.strip() for token in buildings_arg.split(",") if token.strip()]
    return set(names) if names else None


def output_html_path(scenario: str, output_html_arg: str) -> str:
    if output_html_arg.strip():
        return os.path.abspath(output_html_arg.strip())
    return os.path.join(
        os.path.abspath(scenario),
        "outputs",
        "data",
        "solar-radiation",
        "solar_capacity_3d_map.html",
    )


def default_roof_file_path(scenario: str) -> str:
    return os.path.join(
        os.path.abspath(scenario),
        "inputs",
        "building-geometry",
        "roof_surfaces.geojson",
    )


def resolve_roof_file_path(scenario: str, roof_file_arg: str) -> str:
    if roof_file_arg.strip():
        return os.path.abspath(roof_file_arg.strip())

    candidates = [
        default_roof_file_path(scenario),
        os.path.join(
            os.path.abspath(scenario),
            "inputs",
            "building-geometry",
            "roof.geojson",
        ),
        os.path.join(os.path.abspath(scenario), "roof.geojson"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def parse_geojson_crs_name(geojson_obj: dict[str, Any], fallback_crs: str) -> str:
    crs_obj = geojson_obj.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties")
        if isinstance(props, dict):
            name = props.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return fallback_crs


def _iter_polygon_rings(geometry_type: str, coordinates: list[Any]) -> list[list[list[float]]]:
    rings: list[list[list[float]]] = []
    if geometry_type == "Polygon":
        if coordinates and isinstance(coordinates[0], list):
            rings.append(coordinates[0])
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if not isinstance(polygon, list) or not polygon:
                continue
            exterior = polygon[0]
            if isinstance(exterior, list):
                rings.append(exterior)
    return rings


def load_custom_roof_geometries(
    roof_file: str,
    zone_crs: Any,
    building_filter: set[str] | None,
) -> tuple[list[dict[str, Any]], str]:
    if not roof_file.strip():
        return [], ""
    roof_path = os.path.abspath(roof_file)
    if not os.path.exists(roof_path):
        return [], roof_path

    with open(roof_path, "r", encoding="utf-8") as fp:
        try:
            roof_obj = json.load(fp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid roof GeoJSON file: {roof_path}") from exc

    if not isinstance(roof_obj, dict) or roof_obj.get("type") != "FeatureCollection":
        raise ValueError(f"Roof file must be a GeoJSON FeatureCollection: {roof_path}")

    features = roof_obj.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"Roof GeoJSON features are invalid in: {roof_path}")

    zone_crs_name = str(zone_crs) if zone_crs is not None else "EPSG:4326"
    source_crs_name = parse_geojson_crs_name(roof_obj, fallback_crs=zone_crs_name)
    Transformer = _import_pyproj_transformer()
    transformer = Transformer.from_crs(source_crs_name, "EPSG:4326", always_xy=True)

    roofs: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties", {})
        geometry = feature.get("geometry")
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            continue

        building = str(properties.get("building", properties.get("name", ""))).strip()
        if not building:
            continue
        if building_filter is not None and building not in building_filter:
            continue

        geometry_type = str(geometry.get("type", ""))
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list):
            continue

        roof_id = str(properties.get("roof_id", ""))
        rings = _iter_polygon_rings(geometry_type, coordinates)
        for ring_index, ring in enumerate(rings):
            if not isinstance(ring, list) or len(ring) < 3:
                continue

            points_wgs84: list[list[float]] = []
            valid_3d = True
            for point in ring:
                if not isinstance(point, list) or len(point) < 3:
                    valid_3d = False
                    break
                x = float(point[0])
                y = float(point[1])
                z = float(point[2])
                lon, lat = transformer.transform(x, y)
                points_wgs84.append([float(lon), float(lat), z])
            if not valid_3d:
                continue

            ordered_ring = _ordered_ring_from_points_3d(points_wgs84)
            if ordered_ring is None:
                continue

            roof_name = roof_id if roof_id else str(ring_index + 1)
            z_vals = [pt[2] for pt in ordered_ring]
            roofs.append(
                {
                    "building": building,
                    "roof_id": roof_name,
                    "polygon_abs": ordered_ring,
                    "z_min_abs": float(min(z_vals)),
                    "z_max_abs": float(max(z_vals)),
                    "z_mean_abs": float(sum(z_vals) / max(1, len(z_vals))),
                }
            )
    return roofs, roof_path

def load_zone_geometries(locator: InputLocator, building_filter: set[str] | None):
    gpd = _import_geopandas()
    zone_path = locator.get_zone_geometry()
    if not os.path.exists(zone_path):
        raise FileNotFoundError(f"Zone geometry does not exist: {zone_path}")

    zone = gpd.read_file(zone_path)
    if "name" not in zone.columns:
        raise ValueError(f"Zone geometry is missing required 'name' column: {zone_path}")
    zone = zone[~zone.geometry.is_empty & zone.geometry.notna()].copy()

    if building_filter is not None:
        zone = zone[zone["name"].astype(str).isin(building_filter)].copy()
    if zone.empty:
        raise ValueError("No buildings found for the selected filter.")
    return zone


def read_vertical_profile_from_geometry_csv(path: str) -> dict[str, float] | None:
    pd = _import_pandas()
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    required = {"TYPE", "Zcoor", "terrain_elevation"}
    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df["TYPE"] = df["TYPE"].astype(str).str.lower()
    df["Zcoor"] = pd.to_numeric(df["Zcoor"], errors="coerce")
    df["terrain_elevation"] = pd.to_numeric(df["terrain_elevation"], errors="coerce")

    roof = df[df["TYPE"] == "roofs"]["Zcoor"].dropna()
    underside = df[df["TYPE"] == "undersides"]["Zcoor"].dropna()
    terrain = df["terrain_elevation"].dropna()

    if len(roof) == 0:
        return None
    max_roof = float(roof.max())
    min_roof = float(roof.min())
    base_z_abs: float
    height: float
    if len(underside) > 0:
        base_z_abs = float(underside.min())
        height = max_roof - base_z_abs
    elif len(terrain) > 0:
        base_z_abs = float(terrain.iloc[0])
        height = max_roof - base_z_abs
    else:
        base_z_abs = min_roof
        height = max_roof - base_z_abs
    if height < 0:
        return None
    return {
        "building_height_m": float(height),
        "base_z_abs": float(base_z_abs),
        "roof_z_abs_max": float(max_roof),
    }


def fallback_height_from_zone_row(row) -> float:
    for key in ("height_ag", "height", "HEIGHT", "building_height_m"):
        if key in row.index:
            value = to_float(row[key])
            if value is not None and value > 0:
                return float(value)
    for key in ("floors_ag", "floors", "number_of_floors"):
        if key in row.index:
            floors = to_float(row[key])
            if floors is not None and floors > 0:
                return float(floors) * 3.0
    return 10.0


def load_building_vertical_profiles(zone, locator: InputLocator) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for _, row in zone.iterrows():
        building = str(row["name"])
        path = locator.get_radiation_metadata(building)
        parsed = read_vertical_profile_from_geometry_csv(path)
        if parsed is None:
            fallback_height = fallback_height_from_zone_row(row)
            parsed = {
                "building_height_m": float(fallback_height),
                "base_z_abs": 0.0,
                "roof_z_abs_max": float(fallback_height),
            }
        profiles[building] = parsed
    return profiles


def candidate_pv_totals_paths(locator: InputLocator, pv_panel: str) -> list[str]:
    panel_ids = normalise_panel_ids(pv_panel)
    paths: list[str] = []
    for panel in panel_ids:
        panel_no_prefix = panel.replace("PV_", "", 1)
        paths.append(locator.PV_total_buildings(panel_no_prefix))
    panel_token = pv_panel.strip()
    if panel_token:
        manual = os.path.join(locator.solar_potential_folder(), f"PV_{panel_token}_total_buildings.csv")
        if manual not in paths:
            paths.append(manual)
    return paths


def load_solar_capacity_by_building(locator: InputLocator, pv_panel: str) -> tuple[dict[str, float], str]:
    pd = _import_pandas()
    for path in candidate_pv_totals_paths(locator, pv_panel):
        if not os.path.exists(path):
            continue
        try:
            totals = pd.read_csv(path)
        except Exception:
            continue
        if "name" not in totals.columns or "E_PV_gen_kWh" not in totals.columns:
            continue
        totals["name"] = totals["name"].astype(str)
        totals["E_PV_gen_kWh"] = pd.to_numeric(totals["E_PV_gen_kWh"], errors="coerce").fillna(0.0)
        return dict(zip(totals["name"], totals["E_PV_gen_kWh"])), path
    raise FileNotFoundError(
        "Could not find a valid PV totals file. Run photovoltaic first, then retry."
    )


def map_colour(value: float, value_max: float, low: tuple[int, int, int], high: tuple[int, int, int]) -> list[int]:
    if value_max <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / value_max))
    return [
        int(low[0] + ratio * (high[0] - low[0])),
        int(low[1] + ratio * (high[1] - low[1])),
        int(low[2] + ratio * (high[2] - low[2])),
    ]


def build_custom_roofs_payload(
    custom_roofs_abs: list[dict[str, Any]],
    vertical_profiles: dict[str, dict[str, float]],
    solar_by_building_kwh: dict[str, float],
    roof_z_offset_m: float = DEFAULT_CUSTOM_ROOF_Z_OFFSET_M,
) -> list[dict[str, Any]]:
    if not custom_roofs_abs:
        return []

    max_solar_kwh = max(solar_by_building_kwh.values()) if solar_by_building_kwh else 0.0
    roof_fill_low = (255, 222, 182)
    roof_fill_high = (234, 88, 12)
    roof_line_low = (201, 86, 16)
    roof_line_high = (140, 35, 0)

    custom_roofs: list[dict[str, Any]] = []
    for roof in custom_roofs_abs:
        building = str(roof["building"])
        profile = vertical_profiles.get(building, {})
        building_height = float(profile.get("building_height_m", 10.0))
        base_z_abs = profile.get("base_z_abs")

        polygon_abs = roof["polygon_abs"]
        z_values_abs = [float(pt[2]) for pt in polygon_abs]
        z_min_abs = min(z_values_abs)
        z_max_abs = max(z_values_abs)

        if base_z_abs is not None:
            polygon_rel = [[pt[0], pt[1], max(0.0, float(pt[2]) - float(base_z_abs))] for pt in polygon_abs]
        else:
            polygon_rel = [
                [pt[0], pt[1], max(0.0, building_height + (float(pt[2]) - z_min_abs))]
                for pt in polygon_abs
            ]

        z_values_rel = [float(pt[2]) for pt in polygon_rel]
        roof_z_min_rel = float(min(z_values_rel))
        roof_z_max_rel = float(max(z_values_rel))
        # Lift roofs slightly above the building shell to avoid z-fighting / hidden coplanar faces.
        target_min_roof_top = building_height + max(0.0, roof_z_offset_m)
        if roof_z_max_rel < target_min_roof_top:
            lift_delta = target_min_roof_top - roof_z_max_rel
            polygon_rel = [[pt[0], pt[1], float(pt[2]) + lift_delta] for pt in polygon_rel]
            z_values_rel = [float(pt[2]) for pt in polygon_rel]
            roof_z_min_rel = float(min(z_values_rel))
            roof_z_max_rel = float(max(z_values_rel))

        roof_z_mean_rel = float(sum(z_values_rel) / max(1, len(z_values_rel)))
        solar_kwh = float(solar_by_building_kwh.get(building, 0.0))
        fill_colour = map_colour(solar_kwh, max_solar_kwh, roof_fill_low, roof_fill_high)
        line_colour = map_colour(solar_kwh, max_solar_kwh, roof_line_low, roof_line_high)

        custom_roofs.append(
            {
                "building": building,
                "roof_id": str(roof.get("roof_id", "")),
                "polygon": polygon_rel,
                "path": polygon_rel,
                "solar_kwh": solar_kwh,
                "roof_z_min_rel": roof_z_min_rel,
                "roof_z_max_rel": roof_z_max_rel,
                "roof_z_mean_rel": roof_z_mean_rel,
                "roof_span_rel": max(0.0, roof_z_max_rel - roof_z_min_rel),
                "roof_z_min_abs": float(z_min_abs),
                "roof_z_max_abs": float(z_max_abs),
                "fill_colour": fill_colour,
                "line_colour": line_colour,
            }
        )
    return custom_roofs


def build_visualisation_payload(
    zone,
    vertical_profiles: dict[str, dict[str, float]],
    solar_by_building_kwh: dict[str, float],
    min_bar_height_m: float,
    bar_width_m: float,
    custom_roofs_abs: list[dict[str, Any]],
    roof_z_offset_m: float = DEFAULT_CUSTOM_ROOF_Z_OFFSET_M,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    zone = zone.copy()
    zone["name"] = zone["name"].astype(str)
    zone["building_height_m"] = zone["name"].map(
        lambda b: float(vertical_profiles.get(str(b), {}).get("building_height_m", 10.0))
    ).fillna(10.0)
    zone["solar_kwh"] = zone["name"].map(solar_by_building_kwh).fillna(0.0)
    zone["solar_mwh"] = zone["solar_kwh"] / 1000.0

    max_solar = float(zone["solar_kwh"].max()) if len(zone) else 0.0
    max_height = float(zone["building_height_m"].max()) if len(zone) else 0.0

    building_colour_low = (214, 223, 242)
    building_colour_high = (16, 77, 155)
    bar_colour_low = (255, 204, 102)
    bar_colour_high = (226, 71, 23)

    building_fill_colours: list[list[int]] = []
    building_line_colours: list[list[int]] = []
    for _, row in zone.iterrows():
        bcol = map_colour(float(row["solar_kwh"]), max_solar, building_colour_low, building_colour_high)
        lcol = [max(0, c - 35) for c in bcol]
        building_fill_colours.append(bcol)
        building_line_colours.append(lcol)

    zone["fill_colour"] = building_fill_colours
    zone["line_colour"] = building_line_colours

    gpd = _import_geopandas()
    centroids = zone.geometry.centroid
    centres = gpd.GeoDataFrame({"name": zone["name"]}, geometry=centroids, crs=zone.crs)
    centres_wgs84 = centres.to_crs(epsg=4326)
    zone_wgs84 = zone.to_crs(epsg=4326)

    bars: list[dict[str, Any]] = []
    for _, row in zone_wgs84.iterrows():
        building = str(row["name"])
        centroid = centres_wgs84.loc[centres_wgs84["name"] == building].geometry.iloc[0]
        solar_kwh = float(row["solar_kwh"])
        solar_mwh = solar_kwh / 1000.0
        bars.append(
            {
                "name": building,
                "lon": float(centroid.x),
                "lat": float(centroid.y),
                "base_height_m": float(row["building_height_m"]),
                "solar_kwh": solar_kwh,
                "solar_mwh": solar_mwh,
                "bar_height_m_unscaled": max(min_bar_height_m, 1.0) if solar_kwh > 0 else 0.0,
                "bar_width_m": float(bar_width_m),
                "bar_colour": map_colour(solar_kwh, max_solar, bar_colour_low, bar_colour_high),
            }
        )

    centre_lat = float(centres_wgs84.geometry.y.mean())
    centre_lon = float(centres_wgs84.geometry.x.mean())
    metadata = {
        "building_count": int(len(zone_wgs84)),
        "custom_roof_count": int(len(custom_roofs_abs)),
        "max_solar_kwh": max_solar,
        "max_solar_mwh": max_solar / 1000.0,
        "max_building_height_m": max_height,
        "centre_lat": centre_lat,
        "centre_lon": centre_lon,
    }

    custom_roofs = build_custom_roofs_payload(
        custom_roofs_abs=custom_roofs_abs,
        vertical_profiles=vertical_profiles,
        solar_by_building_kwh=solar_by_building_kwh,
        roof_z_offset_m=roof_z_offset_m,
    )

    geojson = json.loads(zone_wgs84.to_json())
    return geojson, bars, custom_roofs, metadata


def render_html(
    output_path: str,
    scenario: str,
    pv_panel: str,
    map_style: str,
    bar_scale_m_per_mwh: float,
    min_bar_height_m: float,
    building_geojson: dict[str, Any],
    bars: list[dict[str, Any]],
    custom_roofs: list[dict[str, Any]],
    metadata: dict[str, Any],
    pv_source_path: str,
    custom_roof_source_path: str,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    payload = {
        "scenario": os.path.abspath(scenario),
        "pv_panel": pv_panel,
        "map_style": map_style,
        "bar_scale_m_per_mwh": bar_scale_m_per_mwh,
        "min_bar_height_m": min_bar_height_m,
        "building_geojson": building_geojson,
        "bars": bars,
        "custom_roofs": custom_roofs,
        "meta": metadata,
        "pv_source_path": pv_source_path,
        "custom_roof_source_path": custom_roof_source_path,
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CEA Solar Capacity 3D Map</title>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <script src="https://unpkg.com/@deck.gl/carto@latest/dist.min.js"></script>
  <script src="https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.css" rel="stylesheet" />
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      background: #f4f8fb;
      color: #1f2a37;
    }}
    #root {{
      display: grid;
      grid-template-columns: minmax(320px, 390px) 1fr;
      width: 100%;
      height: 100%;
    }}
    #sidebar {{
      background: linear-gradient(180deg, #f7fbff 0%, #edf5ff 100%);
      border-right: 1px solid #d6e0ea;
      padding: 16px 14px;
      box-sizing: border-box;
      overflow: auto;
    }}
    #map {{
      position: relative;
      width: 100%;
      height: 100%;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 20px;
      letter-spacing: 0.3px;
    }}
    .meta {{
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 10px;
      color: #334155;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d8e3ef;
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 10px;
    }}
    .panel h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0.2px;
    }}
    .value {{
      font-weight: 600;
      color: #0f4c9c;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .hint {{
      font-size: 12px;
      color: #475569;
    }}
    .layer-row {{
      display: grid;
      grid-template-columns: auto 1fr;
      align-items: center;
      column-gap: 8px;
      margin-bottom: 6px;
      font-size: 13px;
      color: #334155;
    }}
    .layer-row input {{
      width: 14px;
      height: 14px;
      margin: 0;
    }}
    #details {{
      min-height: 100px;
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-line;
    }}
    .legend-scale {{
      font-family: "Consolas", "Courier New", monospace;
      font-size: 12px;
      color: #0f172a;
      background: #f8fbff;
      border: 1px dashed #bcd0e5;
      border-radius: 8px;
      padding: 8px;
    }}
  </style>
</head>
<body>
  <div id="root">
    <aside id="sidebar">
      <h1>Solar Capacity 3D Visualiser</h1>
      <div class="meta">
        Scenario: <span class="value" id="scenarioPath"></span><br/>
        Panel: <span class="value" id="panelId"></span><br/>
        Buildings: <span class="value" id="buildingCount"></span><br/>
        Custom roofs: <span class="value" id="customRoofCount"></span><br/>
      </div>
      <div class="panel">
        <h2>Bar Scale</h2>
        <div class="control-row">
          <label for="scaleRange">Metres per MWh/year</label>
          <span class="value" id="scaleLabel"></span>
        </div>
        <input id="scaleRange" type="range" min="0.002" max="0.25" step="0.001" style="width:100%;" />
        <div class="hint">Adjust bar height scaling without recalculating inputs.</div>
      </div>
      <div class="panel">
        <h2>Layers</h2>
        <label class="layer-row">
          <input id="toggleBuildings" type="checkbox" checked />
          <span>Buildings (3D shell)</span>
        </label>
        <label class="layer-row">
          <input id="toggleCustomRoofs" type="checkbox" checked />
          <span>Custom roofs (3D)</span>
        </label>
        <label class="layer-row">
          <input id="toggleCustomRoofEdges" type="checkbox" checked />
          <span>Custom roof edges</span>
        </label>
        <label class="layer-row">
          <input id="toggleBars" type="checkbox" checked />
          <span>Solar capacity bars</span>
        </label>
      </div>
      <div class="panel">
        <h2>Scale Readout</h2>
        <div class="legend-scale" id="scaleReadout"></div>
      </div>
      <div class="panel">
        <h2>Clicked Bar Details</h2>
        <div id="details">Click a solar-capacity bar to inspect building metrics.</div>
      </div>
      <div class="meta">
        PV totals source:<br/><span id="pvSourcePath"></span>
      </div>
      <div class="meta">
        Custom roof source:<br/><span id="customRoofPath"></span>
      </div>
    </aside>
    <main id="map"></main>
  </div>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=True)};
    const mapContainer = document.getElementById("map");
    const detailsEl = document.getElementById("details");
    const scaleRange = document.getElementById("scaleRange");
    const scaleLabel = document.getElementById("scaleLabel");
    const scaleReadout = document.getElementById("scaleReadout");
    const toggleBuildings = document.getElementById("toggleBuildings");
    const toggleCustomRoofs = document.getElementById("toggleCustomRoofs");
    const toggleCustomRoofEdges = document.getElementById("toggleCustomRoofEdges");
    const toggleBars = document.getElementById("toggleBars");
    document.getElementById("scenarioPath").textContent = payload.scenario;
    document.getElementById("panelId").textContent = payload.pv_panel;
    document.getElementById("buildingCount").textContent = String(payload.meta.building_count);
    document.getElementById("customRoofCount").textContent = String(payload.meta.custom_roof_count || 0);
    document.getElementById("pvSourcePath").textContent = payload.pv_source_path;
    document.getElementById("customRoofPath").textContent = payload.custom_roof_source_path || "(none)";

    let currentScale = Number(payload.bar_scale_m_per_mwh);
    scaleRange.value = String(currentScale);
    const layerVisibility = {{
      buildings: true,
      customRoofs: true,
      customRoofEdges: true,
      bars: true
    }};

    function formatNumber(value, digits = 2) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toLocaleString(undefined, {{maximumFractionDigits: digits}});
    }}

    function updateScaleReadout() {{
      scaleLabel.textContent = currentScale.toFixed(3);
      const oneMetreInMWh = currentScale > 0 ? (1 / currentScale) : 0;
      scaleReadout.textContent =
        `1 m of bar height = ${{formatNumber(oneMetreInMWh, 3)}} MWh/year\\n` +
        `Max building solar capacity = ${{formatNumber(payload.meta.max_solar_mwh, 2)}} MWh/year\\n` +
        `Minimum non-zero bar height = ${{formatNumber(payload.min_bar_height_m, 2)}} m`;
    }}
    updateScaleReadout();

    function toBarRecord(raw) {{
      const scaledHeight = raw.solar_mwh > 0
        ? Math.max(payload.min_bar_height_m, raw.solar_mwh * currentScale)
        : 0;
      return {{
        ...raw,
        bar_height_scaled_m: scaledHeight,
        source_position: [raw.lon, raw.lat, raw.base_height_m],
        target_position: [raw.lon, raw.lat, raw.base_height_m + scaledHeight]
      }};
    }}

    function buildLayers() {{
      const bars = payload.bars.map(toBarRecord);
      const geojsonLayer = new deck.GeoJsonLayer({{
        id: "buildings-3d",
        data: payload.building_geojson,
        pickable: true,
        stroked: true,
        filled: true,
        extruded: true,
        wireframe: false,
        opacity: {DEFAULT_BUILDING_OPACITY / 255.0},
        getFillColor: f => f.properties.fill_colour || [170, 190, 210],
        getLineColor: f => f.properties.line_colour || [80, 100, 120],
        lineWidthMinPixels: 1.0,
        getElevation: f => Number(f.properties.building_height_m || 10),
        onClick: info => {{
          if (!info || !info.object) return;
          const p = info.object.properties || {{}};
          detailsEl.textContent =
            `Building: ${{p.name || "unknown"}}\\n` +
            `Building height: ${{formatNumber(p.building_height_m, 2)}} m\\n` +
            `Solar capacity: ${{formatNumber(p.solar_kwh, 2)}} kWh/year\\n` +
            `Solar capacity: ${{formatNumber((p.solar_kwh || 0) / 1000, 2)}} MWh/year`;
        }}
      }});

      const barsLayer = new deck.LineLayer({{
        id: "solar-bars",
        data: bars,
        pickable: true,
        widthUnits: "meters",
        getWidth: d => d.bar_width_m,
        getSourcePosition: d => d.source_position,
        getTargetPosition: d => d.target_position,
        getColor: d => [...(d.bar_colour || [240, 120, 45]), {DEFAULT_BAR_OPACITY}],
        onClick: info => {{
          if (!info || !info.object) return;
          const d = info.object;
          detailsEl.textContent =
            `Building: ${{d.name}}\\n` +
            `Solar capacity: ${{formatNumber(d.solar_kwh, 2)}} kWh/year\\n` +
            `Solar capacity: ${{formatNumber(d.solar_mwh, 2)}} MWh/year\\n` +
            `Building top elevation: ${{formatNumber(d.base_height_m, 2)}} m\\n` +
            `Bar height (scaled): ${{formatNumber(d.bar_height_scaled_m, 2)}} m\\n` +
            `Scale used: ${{formatNumber(currentScale, 3)}} m per MWh/year`;
        }}
      }});

      const customRoofLayer = new deck.PolygonLayer({{
        id: "custom-roofs",
        data: payload.custom_roofs || [],
        pickable: true,
        stroked: true,
        filled: true,
        wireframe: false,
        extruded: false,
        getPolygon: d => d.polygon,
        getElevation: d => 0,
        getFillColor: d => [...(d.fill_colour || [245, 158, 11]), 180],
        getLineColor: d => d.line_colour || [151, 65, 20],
        lineWidthUnits: "pixels",
        lineWidthMinPixels: 1.6,
        onClick: info => {{
          if (!info || !info.object) return;
          const d = info.object;
          detailsEl.textContent =
            `Building: ${{d.building}}\\n` +
            `Custom roof id: ${{d.roof_id || "-"}}\\n` +
            `Roof elevation range (relative): ${{formatNumber(d.roof_z_min_rel, 2)}} - ${{formatNumber(d.roof_z_max_rel, 2)}} m\\n` +
            `Roof vertical span: ${{formatNumber(d.roof_span_rel, 2)}} m\\n` +
            `Solar capacity (building): ${{formatNumber(d.solar_kwh, 2)}} kWh/year`;
        }}
      }});

      const customRoofEdgeLayer = new deck.PathLayer({{
        id: "custom-roof-edges",
        data: payload.custom_roofs || [],
        pickable: false,
        widthUnits: "pixels",
        getWidth: 2,
        getPath: d => d.path,
        getColor: d => d.line_colour || [122, 45, 10]
      }});

      const layers = [];
      if (layerVisibility.buildings) layers.push(geojsonLayer);
      if (layerVisibility.customRoofs) layers.push(customRoofLayer);
      if (layerVisibility.customRoofEdges) layers.push(customRoofEdgeLayer);
      if (layerVisibility.bars) layers.push(barsLayer);
      return layers;
    }}

    const deckgl = new deck.DeckGL({{
      container: mapContainer,
      mapStyle: payload.map_style,
      initialViewState: {{
        latitude: payload.meta.centre_lat,
        longitude: payload.meta.centre_lon,
        zoom: 16,
        bearing: -20,
        pitch: 50
      }},
      controller: true,
      layers: buildLayers(),
      getTooltip: info => {{
        if (!info || !info.object) return null;
        if (info.layer && info.layer.id === "solar-bars") {{
          return {{
            text:
              `Building: ${{info.object.name}}\\n` +
              `Solar: ${{formatNumber(info.object.solar_kwh, 1)}} kWh/year\\n` +
              `Bar: ${{formatNumber(info.object.bar_height_scaled_m, 2)}} m`
          }};
        }}
        if (info.layer && info.layer.id === "buildings-3d") {{
          const p = info.object.properties || {{}};
          return {{
            text:
              `Building: ${{p.name || "unknown"}}\\n` +
              `Height: ${{formatNumber(p.building_height_m, 2)}} m\\n` +
              `Solar: ${{formatNumber(p.solar_kwh, 1)}} kWh/year`
          }};
        }}
        if (info.layer && info.layer.id === "custom-roofs") {{
          return {{
            text:
              `Custom roof - ${{info.object.building}}\\n` +
              `Roof id: ${{info.object.roof_id || "-"}}\\n` +
              `Roof span: ${{formatNumber(info.object.roof_span_rel, 2)}} m`
          }};
        }}
        return null;
      }}
    }});

    scaleRange.addEventListener("input", evt => {{
      const next = Number(evt.target.value);
      if (!Number.isFinite(next) || next <= 0) return;
      currentScale = next;
      updateScaleReadout();
      deckgl.setProps({{layers: buildLayers()}});
    }});

    function bindLayerToggle(element, key) {{
      if (!element) return;
      layerVisibility[key] = Boolean(element.checked);
      element.addEventListener("change", event => {{
        layerVisibility[key] = Boolean(event.target.checked);
        deckgl.setProps({{layers: buildLayers()}});
      }});
    }}

    bindLayerToggle(toggleBuildings, "buildings");
    bindLayerToggle(toggleCustomRoofs, "customRoofs");
    bindLayerToggle(toggleCustomRoofEdges, "customRoofEdges");
    bindLayerToggle(toggleBars, "bars");
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(html)


def main() -> None:
    args = parse_args()
    if args.bar_scale_m_per_mwh <= 0:
        raise ValueError("--bar-scale-m-per-mwh must be greater than zero.")
    if args.min_bar_height_m < 0:
        raise ValueError("--min-bar-height-m must be non-negative.")
    if args.bar_width_m <= 0:
        raise ValueError("--bar-width-m must be greater than zero.")
    if args.custom_roof_z_offset_m < 0:
        raise ValueError("--custom-roof-z-offset-m must be non-negative.")

    scenario = os.path.abspath(args.scenario)
    if not os.path.isdir(scenario):
        raise NotADirectoryError(f"Scenario folder does not exist: {scenario}")

    locator = InputLocator(scenario)
    building_filter = parse_building_filter(args.buildings)
    zone = load_zone_geometries(locator=locator, building_filter=building_filter)

    vertical_profiles = load_building_vertical_profiles(zone=zone, locator=locator)
    solar_by_building, pv_source_path = load_solar_capacity_by_building(locator=locator, pv_panel=args.pv_panel)
    roof_file = resolve_roof_file_path(scenario, args.roof_file)
    custom_roofs_abs: list[dict[str, Any]] = []
    custom_roof_source_path = ""
    if args.show_custom_roofs:
        custom_roofs_abs, custom_roof_source_path = load_custom_roof_geometries(
            roof_file=roof_file,
            zone_crs=zone.crs,
            building_filter=building_filter,
        )
        if not custom_roofs_abs:
            print(f"[warn] No valid custom roof surfaces were parsed from: {roof_file}")
    else:
        custom_roof_source_path = "(disabled)"

    geojson, bars, custom_roofs, metadata = build_visualisation_payload(
        zone=zone,
        vertical_profiles=vertical_profiles,
        solar_by_building_kwh=solar_by_building,
        min_bar_height_m=args.min_bar_height_m,
        bar_width_m=args.bar_width_m,
        custom_roofs_abs=custom_roofs_abs,
        roof_z_offset_m=args.custom_roof_z_offset_m,
    )

    html_path = output_html_path(scenario, args.output_html)
    render_html(
        output_path=html_path,
        scenario=scenario,
        pv_panel=args.pv_panel,
        map_style=args.map_style,
        bar_scale_m_per_mwh=args.bar_scale_m_per_mwh,
        min_bar_height_m=args.min_bar_height_m,
        building_geojson=geojson,
        bars=bars,
        custom_roofs=custom_roofs,
        metadata=metadata,
        pv_source_path=pv_source_path,
        custom_roof_source_path=custom_roof_source_path,
    )

    print("Solar capacity 3D map visualiser generated successfully.")
    print(f"Scenario: {scenario}")
    print(f"Buildings visualised: {metadata['building_count']}")
    print(f"Custom roofs visualised: {metadata['custom_roof_count']}")
    print(f"Custom roof source: {custom_roof_source_path}")
    print(f"PV totals source: {pv_source_path}")
    print(f"Output HTML: {html_path}")


if __name__ == "__main__":
    main()
