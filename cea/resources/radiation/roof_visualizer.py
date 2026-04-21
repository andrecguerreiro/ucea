"""Minimal local-data 3D visualizer for UCEA roof outputs.

The visualizer is intentionally offline by default: it reads CEA zone geometry,
UCEA roof surfaces, and optional PV totals from the scenario. It does not fetch
or embed satellite imagery.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

from cea.inputlocator import InputLocator
from pyproj import Transformer


DEFAULT_MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
DEFAULT_ROOF_FILE = os.path.join("inputs", "building-geometry", "roof_surfaces.geojson")
DEFAULT_OUTPUT_HTML = os.path.join("outputs", "data", "solar-radiation", "roof_3d_layers_map.html")
DEFAULT_PV_PANEL = "PV1"
DEFAULT_ZONE_HEIGHT_M = 8.0
DEFAULT_BAR_SCALE_M_PER_MWH = 0.15
DEFAULT_MIN_BAR_HEIGHT_M = 1.5
DEFAULT_BAR_WIDTH_M = 9.0
DEFAULT_BAR_BASE_OFFSET_M = 0.4
DEFAULT_COORDINATE_PRECISION = 7
DEFAULT_FLAT_THRESHOLD_DEG = 10.0
ORIENTATION_ORDER = ("flat", "N", "E", "S", "W")


def _import_geopandas():
    try:
        import geopandas as gpd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'geopandas'.") from exc
    return gpd


def _import_pandas():
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'pandas'.") from exc
    return pd


def _import_shapely():
    try:
        from shapely.geometry import mapping  # type: ignore
        from shapely.ops import unary_union  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing dependency 'shapely'.") from exc
    return mapping, unary_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local 3D roof visualizer HTML.")
    parser.add_argument("--scenario", required=True, help="Path to the CEA scenario.")
    parser.add_argument("--roof-file", default="", help="Roof surfaces GeoJSON path.")
    parser.add_argument("--output-html", default="", help="Output HTML path.")
    parser.add_argument("--map-style", default=DEFAULT_MAP_STYLE, help="MapLibre style URL.")
    parser.add_argument("--pv-panel", default=DEFAULT_PV_PANEL, help="PV panel id for PV totals CSV.")
    parser.add_argument(
        "--bar-scale-m-per-mwh",
        type=float,
        default=DEFAULT_BAR_SCALE_M_PER_MWH,
        help="Solar bar height scale in metres per MWh/year.",
    )
    parser.add_argument(
        "--min-bar-height-m",
        type=float,
        default=DEFAULT_MIN_BAR_HEIGHT_M,
        help="Minimum visible solar bar height for non-zero PV.",
    )
    parser.add_argument(
        "--bar-width-m",
        type=float,
        default=DEFAULT_BAR_WIDTH_M,
        help="Solar bar width in metres.",
    )
    parser.add_argument(
        "--coordinate-precision",
        type=int,
        default=DEFAULT_COORDINATE_PRECISION,
        help="Decimal places for lon/lat coordinates embedded in HTML.",
    )
    parser.add_argument(
        "--flat-threshold-deg",
        type=float,
        default=DEFAULT_FLAT_THRESHOLD_DEG,
        help="Tilt threshold (degrees) below which roofs are classified as flat.",
    )
    return parser.parse_args()


def _default_roof_file_path(scenario: str) -> str:
    return os.path.join(os.path.abspath(scenario), DEFAULT_ROOF_FILE)


def _default_output_html_path(scenario: str) -> str:
    return os.path.join(os.path.abspath(scenario), DEFAULT_OUTPUT_HTML)


def _resolve_path(default_path: str, cli_path: str) -> str:
    return os.path.abspath(cli_path.strip()) if cli_path.strip() else default_path


def _geojson_crs_name(geojson_obj: dict[str, Any], fallback_crs: str) -> str:
    crs_obj = geojson_obj.get("crs")
    if isinstance(crs_obj, dict):
        name = crs_obj.get("properties", {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return fallback_crs


def _iter_exterior_rings(geometry: dict[str, Any]):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return
    if geometry_type == "Polygon" and coordinates:
        yield coordinates[0]
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if isinstance(polygon, list) and polygon:
                yield polygon[0]


def _round_number(value: Any, precision: int) -> float:
    return round(float(value), precision)


def _round_coordinates(value: Any, precision: int) -> Any:
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return [_round_number(item, precision) for item in value]
        return [_round_coordinates(item, precision) for item in value]
    return value


def _close_ring(ring: list[list[float]]) -> list[list[float]]:
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _map_colour(value: float, value_max: float, low: tuple[int, int, int], high: tuple[int, int, int]) -> list[int]:
    ratio = 0.0 if value_max <= 0 else max(0.0, min(1.0, value / value_max))
    return [
        int(low[0] + ratio * (high[0] - low[0])),
        int(low[1] + ratio * (high[1] - low[1])),
        int(low[2] + ratio * (high[2] - low[2])),
    ]


def _first_positive_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(str(value).replace(",", "."))
        except ValueError:
            continue
        if number > 0 and math.isfinite(number):
            return number
    return None


def _to_non_negative_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(0.0, number)


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _empty_orientation_area_dict() -> dict[str, float]:
    return {key: 0.0 for key in ORIENTATION_ORDER}


def _rounded_orientation_area_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(values.get(key, 0.0)), 2) for key in ORIENTATION_ORDER}


def _normal_and_area(points_xyz: list[list[float]]) -> tuple[float, float, float, float] | None:
    if len(points_xyz) < 3:
        return None

    points = points_xyz
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return None

    nx = 0.0
    ny = 0.0
    nz = 0.0
    count = len(points)
    for i in range(count):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % count]
        nx += (y1 - y2) * (z1 + z2)
        ny += (z1 - z2) * (x1 + x2)
        nz += (x1 - x2) * (y1 + y2)

    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 1e-9:
        return None
    area = 0.5 * norm
    return nx, ny, nz, area


def _normal_azimuth_deg(nx: float, ny: float) -> float | None:
    horizontal = math.hypot(nx, ny)
    if horizontal <= 1e-9:
        return None
    return (math.degrees(math.atan2(nx, ny)) + 360.0) % 360.0


def _tilt_deg_from_normal(nx: float, ny: float, nz: float) -> float:
    horizontal = math.hypot(nx, ny)
    return math.degrees(math.atan2(horizontal, abs(nz)))


def _orientation_bucket(tilt_deg: float, azimuth_deg: float | None, flat_threshold_deg: float) -> str:
    if tilt_deg <= flat_threshold_deg or azimuth_deg is None:
        return "flat"
    if 45.0 <= azimuth_deg < 135.0:
        return "E"
    if 135.0 <= azimuth_deg < 225.0:
        return "S"
    if 225.0 <= azimuth_deg < 315.0:
        return "W"
    return "N"


def _load_roofs(
    roof_file_path: str,
    precision: int,
    flat_threshold_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, int]]:
    if not os.path.exists(roof_file_path):
        raise FileNotFoundError(f"Roof surfaces file not found: {roof_file_path}")

    with open(roof_file_path, "r", encoding="utf-8") as fp:
        roof_obj = json.load(fp)
    if roof_obj.get("type") != "FeatureCollection":
        raise ValueError(f"Roof file is not a FeatureCollection: {roof_file_path}")

    transformer = Transformer.from_crs(_geojson_crs_name(roof_obj, "EPSG:32629"), "EPSG:4326", always_xy=True)
    raw_roofs: list[dict[str, Any]] = []
    roof_stats: dict[str, dict[str, float]] = {}
    roof_quality = {
        "features_total": 0,
        "rings_total": 0,
        "valid_roofs": 0,
        "ignored_roofs": 0,
        "ignored_missing_building": 0,
        "ignored_invalid_geometry": 0,
        "ignored_zero_area": 0,
    }

    for feature in roof_obj.get("features", []):
        roof_quality["features_total"] += 1
        if not isinstance(feature, dict):
            roof_quality["ignored_roofs"] += 1
            roof_quality["ignored_invalid_geometry"] += 1
            continue

        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        if not isinstance(properties, dict) or not isinstance(geometry, dict):
            roof_quality["ignored_roofs"] += 1
            roof_quality["ignored_invalid_geometry"] += 1
            continue

        building = str(properties.get("building", "")).strip()
        roof_id = str(properties.get("roof_id", "")).strip()
        if not building:
            roof_quality["ignored_roofs"] += 1
            roof_quality["ignored_missing_building"] += 1
            continue

        for source_ring in _iter_exterior_rings(geometry) or []:
            roof_quality["rings_total"] += 1
            ring_xyz: list[list[float]] = []
            ring_wgs84: list[list[float]] = []
            invalid_point_found = False

            for point in source_ring:
                if not isinstance(point, (list, tuple)) or len(point) < 3:
                    invalid_point_found = True
                    continue
                try:
                    x = float(point[0])
                    y = float(point[1])
                    z = float(point[2])
                except (TypeError, ValueError):
                    invalid_point_found = True
                    continue
                if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                    invalid_point_found = True
                    continue
                ring_xyz.append([x, y, z])
                lon, lat = transformer.transform(x, y)
                ring_wgs84.append([_round_number(lon, precision), _round_number(lat, precision), round(z, 2)])

            if len(ring_xyz) < 3 or len(ring_wgs84) < 3 or invalid_point_found:
                roof_quality["ignored_roofs"] += 1
                roof_quality["ignored_invalid_geometry"] += 1
                continue

            ring_xyz = _close_ring(ring_xyz)
            ring_wgs84 = _close_ring(ring_wgs84)
            normal_and_area = _normal_and_area(ring_xyz)
            if normal_and_area is None:
                roof_quality["ignored_roofs"] += 1
                roof_quality["ignored_invalid_geometry"] += 1
                continue

            nx, ny, nz, area_m2 = normal_and_area
            if area_m2 <= 1e-9:
                roof_quality["ignored_roofs"] += 1
                roof_quality["ignored_zero_area"] += 1
                continue

            tilt_deg = _tilt_deg_from_normal(nx, ny, nz)
            azimuth_deg = _normal_azimuth_deg(nx, ny)
            orientation = _orientation_bucket(tilt_deg, azimuth_deg, flat_threshold_deg)

            z_values = [point[2] for point in ring_xyz]
            z_min = float(min(z_values))
            z_max = float(max(z_values))

            raw_roofs.append(
                {
                    "building": building,
                    "roof_id": roof_id,
                    "polygon": ring_wgs84,
                    "z_min": round(z_min, 2),
                    "z_max": round(z_max, 2),
                    "area_m2": round(area_m2, 2),
                    "tilt_deg": round(tilt_deg, 2),
                    "azimuth_deg": None if azimuth_deg is None else round(azimuth_deg, 2),
                    "orientation": orientation,
                }
            )

            stats = roof_stats.setdefault(
                building,
                {
                    "lon_sum": 0.0,
                    "lat_sum": 0.0,
                    "count": 0.0,
                    "z_min": z_min,
                    "z_top": z_max,
                    "roof_count": 0.0,
                    "available_roof_area_m2": 0.0,
                    "orientation_area_flat_m2": 0.0,
                    "orientation_area_N_m2": 0.0,
                    "orientation_area_E_m2": 0.0,
                    "orientation_area_S_m2": 0.0,
                    "orientation_area_W_m2": 0.0,
                },
            )

            for lon, lat, _ in ring_wgs84[:-1]:
                stats["lon_sum"] += lon
                stats["lat_sum"] += lat
                stats["count"] += 1.0
            stats["z_min"] = min(stats["z_min"], z_min)
            stats["z_top"] = max(stats["z_top"], z_max)
            stats["roof_count"] += 1.0
            stats["available_roof_area_m2"] += area_m2
            stats[f"orientation_area_{orientation}_m2"] += area_m2
            roof_quality["valid_roofs"] += 1

    max_z = max((roof["z_max"] for roof in raw_roofs), default=DEFAULT_ZONE_HEIGHT_M)
    for roof in raw_roofs:
        ratio = 0.0 if max_z <= 0 else max(0.0, min(1.0, float(roof["z_max"]) / max_z))
        roof["fill_colour"] = [int(220 + 35 * ratio), int(90 + 70 * ratio), 40, 220]

    return raw_roofs, roof_stats, roof_quality


def _load_zone_payload(
    locator: InputLocator,
    roof_stats: dict[str, dict[str, float]],
    precision: int,
) -> tuple[
    dict[str, Any],
    dict[str, list[float]],
    tuple[float, float],
    tuple[float, float, float, float],
    dict[str, float],
    set[str],
]:
    gpd = _import_geopandas()
    mapping, unary_union = _import_shapely()

    zone_path = locator.get_zone_geometry()
    if not os.path.exists(zone_path):
        raise FileNotFoundError(f"Zone geometry not found: {zone_path}")

    zone = gpd.read_file(zone_path)
    if "name" not in zone.columns:
        raise ValueError("Zone geometry must contain a 'name' column.")
    zone = zone[~zone.geometry.is_empty & zone.geometry.notna()].copy()
    if zone.empty:
        raise ValueError("Zone geometry is empty.")

    zone_wgs84 = zone.to_crs(epsg=4326)
    site = unary_union(zone_wgs84.geometry.values.tolist())
    minx, miny, maxx, maxy = site.bounds
    centre = site.centroid

    features: list[dict[str, Any]] = []
    centroids: dict[str, list[float]] = {}
    heights_by_building: dict[str, float] = {}
    zone_buildings: set[str] = set()

    for row in zone_wgs84.itertuples(index=False):
        row_data = row._asdict()
        name = str(row_data.get("name", "")).strip()
        if name:
            zone_buildings.add(name)

        stats = roof_stats.get(name)
        height = float(stats["z_min"]) if stats else None
        if height is None:
            height = _first_positive_number(row_data.get("height_ag"), row_data.get("height"))
        if height is None:
            floors = _first_positive_number(row_data.get("floors_ag"), row_data.get("floors"))
            height = None if floors is None else floors * 3.0
        if height is None:
            height = DEFAULT_ZONE_HEIGHT_M
        viz_height_m = round(max(3.0, float(height)), 2)
        if name:
            heights_by_building[name] = viz_height_m

        centroid = row_data["geometry"].centroid
        if name:
            centroids[name] = [_round_number(centroid.x, precision), _round_number(centroid.y, precision)]

        geometry = mapping(row_data["geometry"])
        geometry["coordinates"] = _round_coordinates(geometry["coordinates"], precision)
        features.append(
            {
                "type": "Feature",
                "properties": {"name": name, "viz_height_m": viz_height_m},
                "geometry": geometry,
            }
        )

    return (
        {"type": "FeatureCollection", "features": features},
        centroids,
        (float(centre.y), float(centre.x)),
        (float(minx), float(miny), float(maxx), float(maxy)),
        heights_by_building,
        zone_buildings,
    )


def _candidate_pv_totals_paths(locator: InputLocator, pv_panel: str) -> list[str]:
    panel = str(pv_panel).strip() or DEFAULT_PV_PANEL
    panel_no_prefix = panel[3:] if panel.upper().startswith("PV_") else panel
    candidates = [locator.PV_total_buildings(panel_no_prefix)]
    manual_path = os.path.join(locator.solar_potential_folder(), f"PV_{panel_no_prefix}_total_buildings.csv")
    if manual_path not in candidates:
        candidates.append(manual_path)
    return candidates


def _candidate_panel_codes(pv_panel: str) -> list[str]:
    token = str(pv_panel).strip()
    if not token:
        token = DEFAULT_PV_PANEL
    base = token[3:] if token.upper().startswith("PV_") else token
    candidates: list[str] = []
    seen: set[str] = set()
    for value in (token, base, f"PV_{base}"):
        value = value.strip()
        upper = value.upper()
        if value and upper not in seen:
            seen.add(upper)
            candidates.append(value)
    return candidates


def _candidate_panel_placement_paths(locator: InputLocator) -> list[str]:
    scenario = os.path.abspath(locator.scenario)
    search_roots = [
        os.path.join(scenario, "outputs", "data", "solar-radiation", "workflow1_metrics"),
        os.path.join(scenario, "outputs", "data", "roof-workflow-comparison"),
        os.path.join(scenario, "outputs", "data", "solar-radiation"),
    ]

    candidates: list[str] = []
    seen: set[str] = set()
    for root in search_roots:
        base_path = os.path.abspath(os.path.join(root, "panel_placement_summary.csv"))
        if base_path not in seen:
            seen.add(base_path)
            candidates.append(base_path)

        pattern = os.path.join(root, "panel_placement_summary*.csv")
        matches = sorted(
            (os.path.abspath(path) for path in glob.glob(pattern) if os.path.isfile(path)),
            key=lambda path: (0 if os.path.basename(path).lower() == "panel_placement_summary.csv" else 1, -os.path.getmtime(path)),
        )
        for path in matches:
            if path not in seen:
                seen.add(path)
                candidates.append(path)
    return candidates


def _normalise_panel_direction(value: Any) -> str | None:
    token = str(value).strip().lower()
    if not token:
        return None
    token = token.replace("_", "").replace("-", "").replace(" ", "")
    if token in {"flat", "top", "roof", "roofs", "rooftop", "horizontal", "h"}:
        return "flat"
    if token in {"n", "north"}:
        return "N"
    if token in {"e", "east"}:
        return "E"
    if token in {"s", "south"}:
        return "S"
    if token in {"w", "west"}:
        return "W"
    return None


def _select_preferred_workflow_id(workflow_ids: set[str]) -> str:
    cleaned = {str(value).strip() for value in workflow_ids if str(value).strip()}
    if not cleaned:
        return ""

    upper_to_original = {value.upper(): value for value in sorted(cleaned)}
    for preferred in ("WF1", "WORKFLOW1", "W1", "WF0", "WORKFLOW0", "W0"):
        if preferred in upper_to_original:
            return upper_to_original[preferred]
    return sorted(cleaned)[0]


def _load_panel_placement_orientation_by_building(
    locator: InputLocator,
    pv_panel: str,
) -> tuple[dict[str, dict[str, float]], dict[str, float], str]:
    pd = _import_pandas()
    panel_codes = {value.upper() for value in _candidate_panel_codes(pv_panel)}

    for path in _candidate_panel_placement_paths(locator):
        if not os.path.exists(path):
            continue
        try:
            panel_df = pd.read_csv(path)
        except Exception:
            continue
        if panel_df is None or panel_df.empty:
            continue

        panel_df = panel_df.copy()
        panel_df.columns = [str(column).strip() for column in panel_df.columns]
        required_columns = {"building", "direction", "panel_area_m2"}
        if not required_columns.issubset(panel_df.columns):
            continue

        if "missing_status" in panel_df.columns:
            status = panel_df["missing_status"].astype(str).str.strip().str.lower()
            panel_df = panel_df[(status == "") | (status == "ok")]

        panel_df["building"] = panel_df["building"].astype(str).str.strip()
        panel_df["panel_area_m2"] = pd.to_numeric(panel_df["panel_area_m2"], errors="coerce").fillna(0.0)

        if "pv_panel" in panel_df.columns:
            panel_df["pv_panel"] = panel_df["pv_panel"].astype(str).str.strip()
            panel_df = panel_df[panel_df["pv_panel"].str.upper().isin(panel_codes)]

        if panel_df.empty:
            continue

        if "workflow_id" in panel_df.columns:
            workflow_ids = {
                str(value).strip()
                for value in panel_df["workflow_id"].astype(str).tolist()
                if str(value).strip()
            }
            preferred_workflow = _select_preferred_workflow_id(workflow_ids)
            if preferred_workflow:
                panel_df = panel_df[
                    panel_df["workflow_id"].astype(str).str.strip().str.upper() == preferred_workflow.upper()
                ]
        if panel_df.empty:
            continue

        panel_df["direction_key"] = panel_df["direction"].map(_normalise_panel_direction)
        panel_df = panel_df[
            (panel_df["building"] != "")
            & panel_df["direction_key"].notna()
            & (panel_df["panel_area_m2"] > 0)
        ]
        if panel_df.empty:
            continue

        has_generation = "E_PV_gen_kWh_approx" in panel_df.columns
        if has_generation:
            panel_df["E_PV_gen_kWh_approx"] = pd.to_numeric(panel_df["E_PV_gen_kWh_approx"], errors="coerce").fillna(0.0)

        orientation_by_building: dict[str, dict[str, float]] = {}
        generation_by_building: dict[str, float] = {}
        for _, row in panel_df.iterrows():
            building = str(row["building"]).strip()
            direction = _normalise_panel_direction(row["direction_key"])
            if direction is None:
                continue
            area_m2 = _to_non_negative_float(row["panel_area_m2"])
            if area_m2 <= 0:
                continue

            direction_map = orientation_by_building.setdefault(building, _empty_orientation_area_dict())
            direction_map[direction] += area_m2
            if has_generation:
                generation_by_building[building] = generation_by_building.get(building, 0.0) + _to_non_negative_float(
                    row["E_PV_gen_kWh_approx"]
                )

        if orientation_by_building:
            return orientation_by_building, generation_by_building, path

    return {}, {}, ""


def _merge_panel_placement_orientation(
    by_building: dict[str, dict[str, Any]],
    orientation_by_building: dict[str, dict[str, float]],
    generation_by_building: dict[str, float],
) -> None:
    for building, orientation_values in orientation_by_building.items():
        orientation_installed = _empty_orientation_area_dict()
        for key in ORIENTATION_ORDER:
            orientation_installed[key] = _to_non_negative_float(orientation_values.get(key, 0.0))

        orientation_total = sum(orientation_installed.values())
        row = by_building.setdefault(
            building,
            {
                "E_PV_gen_kWh": 0.0,
                "area_PV_m2": 0.0,
                "orientation_area_installed_m2": _empty_orientation_area_dict(),
            },
        )
        row["orientation_area_installed_m2"] = orientation_installed
        if _to_non_negative_float(row.get("area_PV_m2", 0.0)) <= 0 and orientation_total > 0:
            row["area_PV_m2"] = orientation_total
        if _to_non_negative_float(row.get("E_PV_gen_kWh", 0.0)) <= 0:
            generation_kwh = _to_non_negative_float(generation_by_building.get(building, 0.0))
            if generation_kwh > 0:
                row["E_PV_gen_kWh"] = generation_kwh


def _load_pv_metrics_by_building(locator: InputLocator, pv_panel: str) -> tuple[dict[str, dict[str, Any]], str]:
    pd = _import_pandas()
    panel_orientation_by_building, panel_generation_by_building, panel_placement_path = (
        _load_panel_placement_orientation_by_building(locator, pv_panel)
    )

    for path in _candidate_pv_totals_paths(locator, pv_panel):
        if not os.path.exists(path):
            continue
        try:
            totals = pd.read_csv(path)
        except Exception:
            continue
        if "name" not in totals.columns or "E_PV_gen_kWh" not in totals.columns:
            continue

        totals = totals.copy()
        totals["name"] = totals["name"].astype(str).str.strip()
        for column in (
            "E_PV_gen_kWh",
            "area_PV_m2",
            "PV_roofs_top_m2",
            "PV_walls_north_m2",
            "PV_walls_east_m2",
            "PV_walls_south_m2",
            "PV_walls_west_m2",
        ):
            if column not in totals.columns:
                totals[column] = 0.0
            totals[column] = pd.to_numeric(totals[column], errors="coerce").fillna(0.0)

        by_building: dict[str, dict[str, Any]] = {}
        for row in totals.itertuples(index=False):
            building = str(getattr(row, "name", "")).strip()
            if not building:
                continue
            orientation_installed = {
                "flat": _to_non_negative_float(getattr(row, "PV_roofs_top_m2", 0.0)),
                "N": _to_non_negative_float(getattr(row, "PV_walls_north_m2", 0.0)),
                "E": _to_non_negative_float(getattr(row, "PV_walls_east_m2", 0.0)),
                "S": _to_non_negative_float(getattr(row, "PV_walls_south_m2", 0.0)),
                "W": _to_non_negative_float(getattr(row, "PV_walls_west_m2", 0.0)),
            }
            by_building[building] = {
                "E_PV_gen_kWh": _to_non_negative_float(getattr(row, "E_PV_gen_kWh", 0.0)),
                "area_PV_m2": _to_non_negative_float(getattr(row, "area_PV_m2", 0.0)),
                "orientation_area_installed_m2": orientation_installed,
            }
        _merge_panel_placement_orientation(by_building, panel_orientation_by_building, panel_generation_by_building)
        return by_building, path

    if panel_orientation_by_building:
        by_building: dict[str, dict[str, Any]] = {}
        _merge_panel_placement_orientation(by_building, panel_orientation_by_building, panel_generation_by_building)
        return by_building, panel_placement_path

    return {}, ""


def _load_pv_panel_efficiency(locator: InputLocator, pv_panel: str) -> tuple[float | None, str]:
    pd = _import_pandas()
    db_path = locator.get_db4_components_conversion_conversion_technology_csv("PHOTOVOLTAIC_PANELS")
    if not os.path.exists(db_path):
        return None, ""
    try:
        panels = pd.read_csv(db_path)
    except Exception:
        return None, ""
    if "code" not in panels.columns or "PV_n" not in panels.columns:
        return None, ""

    panels = panels.copy()
    panels["code"] = panels["code"].astype(str).str.strip()
    panels["PV_n"] = pd.to_numeric(panels["PV_n"], errors="coerce")
    code_lookup: dict[str, tuple[str, float]] = {}
    for row in panels.itertuples(index=False):
        code = str(getattr(row, "code", "")).strip()
        if not code:
            continue
        pv_n = getattr(row, "PV_n", float("nan"))
        try:
            efficiency = float(pv_n)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(efficiency) or efficiency <= 0:
            continue
        code_lookup[code.upper()] = (code, efficiency)

    for candidate in _candidate_panel_codes(pv_panel):
        hit = code_lookup.get(candidate.upper())
        if hit is not None:
            return hit[1], hit[0]
    return None, ""


def _build_solar_bars(
    roof_stats: dict[str, dict[str, float]],
    zone_centroids: dict[str, list[float]],
    zone_heights_by_building: dict[str, float],
    solar_by_building_kwh: dict[str, float],
    bar_scale_m_per_mwh: float,
    min_bar_height_m: float,
    bar_width_m: float,
    precision: int,
) -> list[dict[str, Any]]:
    max_solar_kwh = max((float(value) for value in solar_by_building_kwh.values()), default=0.0)
    bars: list[dict[str, Any]] = []
    for building, solar_kwh_raw in solar_by_building_kwh.items():
        solar_kwh = float(solar_kwh_raw)
        solar_mwh = solar_kwh / 1000.0
        stats = roof_stats.get(str(building))
        if solar_mwh <= 0.0:
            continue

        if stats and stats["count"] > 0:
            lon = _round_number(stats["lon_sum"] / stats["count"], precision)
            lat = _round_number(stats["lat_sum"] / stats["count"], precision)
            base_height = round(float(stats["z_top"]) + DEFAULT_BAR_BASE_OFFSET_M, 2)
        else:
            centroid = zone_centroids.get(str(building))
            if centroid is None:
                continue
            lon = float(centroid[0])
            lat = float(centroid[1])
            base_height = round(
                float(zone_heights_by_building.get(str(building), DEFAULT_ZONE_HEIGHT_M)) + DEFAULT_BAR_BASE_OFFSET_M,
                2,
            )

        bar_height = round(max(min_bar_height_m, solar_mwh * bar_scale_m_per_mwh), 2)
        bars.append(
            {
                "building": str(building),
                "solar_kwh": round(solar_kwh, 2),
                "solar_mwh": round(solar_mwh, 2),
                "base_height_m": base_height,
                "bar_height_m": bar_height,
                "bar_width_m": float(bar_width_m),
                "source_position": [lon, lat, base_height],
                "target_position": [lon, lat, round(base_height + bar_height, 2)],
                "bar_colour": _map_colour(solar_kwh, max_solar_kwh, (255, 204, 102), (226, 71, 23)),
            }
        )
    return bars


def _build_decision_metrics(
    zone_buildings: set[str],
    zone_heights_by_building: dict[str, float],
    roof_stats: dict[str, dict[str, float]],
    roof_quality: dict[str, int],
    pv_metrics_by_building: dict[str, dict[str, Any]],
    pv_source_path: str,
    pv_panel_efficiency: float | None,
    pv_panel_code: str,
    roof_availability_by_building: dict[str, dict[str, float]] | None = None,
    roof_availability_source_path: str = "",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    total_zone_buildings = len(zone_buildings)
    custom_roof_buildings = {name for name in roof_stats if name in zone_buildings}
    panel_availability_by_building = roof_availability_by_building or {}

    available_orientation_totals = _empty_orientation_area_dict()
    installed_orientation_totals = _empty_orientation_area_dict()
    building_metrics: dict[str, dict[str, Any]] = {}

    total_available_roof_area_m2 = 0.0
    total_pv_generation_kwh = 0.0
    total_pv_area_m2 = 0.0
    pv_rows_in_zone = 0
    buildings_with_pv_positive = 0

    for building in sorted(zone_buildings):
        height_m = float(zone_heights_by_building.get(building, DEFAULT_ZONE_HEIGHT_M))
        roof_info = roof_stats.get(building, {})
        roof_count = int(round(float(roof_info.get("roof_count", 0.0))))

        panel_available_by_orientation = panel_availability_by_building.get(building)
        panel_available_by_orientation = (
            {
                key: _to_non_negative_float(panel_available_by_orientation.get(key, 0.0))
                for key in ORIENTATION_ORDER
            }
            if isinstance(panel_available_by_orientation, dict)
            else None
        )
        if panel_available_by_orientation and sum(panel_available_by_orientation.values()) > 0:
            available_by_orientation = panel_available_by_orientation
            available_roof_area_m2 = sum(available_by_orientation.values())
            availability_source = "panel_placement_summary"
        else:
            available_by_orientation = {
                key: float(roof_info.get(f"orientation_area_{key}_m2", 0.0))
                for key in ORIENTATION_ORDER
            }
            available_roof_area_m2 = float(roof_info.get("available_roof_area_m2", 0.0))
            availability_source = "roof_surfaces_geojson"

        if available_roof_area_m2 > 0 and (
            availability_source == "panel_placement_summary" or building in custom_roof_buildings
        ):
            total_available_roof_area_m2 += available_roof_area_m2
            for key in ORIENTATION_ORDER:
                available_orientation_totals[key] += available_by_orientation[key]

        pv_row = pv_metrics_by_building.get(building)
        if pv_row is not None:
            pv_rows_in_zone += 1

        pv_generation_kwh = float(pv_row["E_PV_gen_kWh"]) if pv_row else 0.0
        pv_area_m2 = float(pv_row["area_PV_m2"]) if pv_row else 0.0
        if pv_row:
            pv_installed_by_orientation = {
                key: float((pv_row.get("orientation_area_installed_m2", {}) or {}).get(key, 0.0))
                for key in ORIENTATION_ORDER
            }
        else:
            pv_installed_by_orientation = _empty_orientation_area_dict()
        pv_installed_orientation_total = sum(pv_installed_by_orientation.values())

        total_pv_generation_kwh += pv_generation_kwh
        total_pv_area_m2 += pv_area_m2
        for key in ORIENTATION_ORDER:
            installed_orientation_totals[key] += pv_installed_by_orientation[key]

        if pv_generation_kwh > 0:
            buildings_with_pv_positive += 1

        installed_power_kwp = None if pv_panel_efficiency is None else pv_area_m2 * pv_panel_efficiency

        building_metrics[building] = {
            "height_m": round(height_m, 2),
            "custom_roof_detected": building in custom_roof_buildings,
            "custom_roof_count": roof_count,
            "roof_availability_source": availability_source,
            "available_roof_area_m2": round(available_roof_area_m2, 2),
            "available_by_orientation_m2": _rounded_orientation_area_dict(available_by_orientation),
            "pv_row_present": pv_row is not None,
            "pv_generation_kwh": round(pv_generation_kwh, 2),
            "pv_installed_area_m2": round(pv_area_m2, 2),
            "pv_installed_power_kwp": _round_or_none(installed_power_kwp, 3),
            "pv_installed_by_orientation_m2": _rounded_orientation_area_dict(pv_installed_by_orientation),
            "pv_installed_orientation_total_m2": round(pv_installed_orientation_total, 2),
        }

    pv_rows_not_in_zone = sum(1 for building in pv_metrics_by_building if building not in zone_buildings)
    missing_pv_rows_for_zone = max(0, total_zone_buildings - pv_rows_in_zone)

    total_pv_installed_power_kwp = None if pv_panel_efficiency is None else total_pv_area_m2 * pv_panel_efficiency
    kwh_per_installed_m2 = None if total_pv_area_m2 <= 0 else total_pv_generation_kwh / total_pv_area_m2
    kwh_per_available_roof_m2 = (
        None if total_available_roof_area_m2 <= 0 else total_pv_generation_kwh / total_available_roof_area_m2
    )
    roof_utilisation_ratio = None if total_available_roof_area_m2 <= 0 else total_pv_area_m2 / total_available_roof_area_m2

    available_orientation_share = {
        key: None if total_available_roof_area_m2 <= 0 else available_orientation_totals[key] / total_available_roof_area_m2
        for key in ORIENTATION_ORDER
    }
    installed_orientation_share = {
        key: None if total_pv_area_m2 <= 0 else installed_orientation_totals[key] / total_pv_area_m2
        for key in ORIENTATION_ORDER
    }

    kpis = {
        "totals": {
            "pv_area_m2": round(total_pv_area_m2, 2),
            "pv_generation_kwh": round(total_pv_generation_kwh, 2),
            "pv_generation_mwh": round(total_pv_generation_kwh / 1000.0, 2),
            "installed_power_kwp": _round_or_none(total_pv_installed_power_kwp, 3),
        },
        "coverage": {
            "total_zone_buildings": int(total_zone_buildings),
            "custom_roof_buildings": int(len(custom_roof_buildings)),
            "custom_roof_share": None if total_zone_buildings <= 0 else len(custom_roof_buildings) / total_zone_buildings,
            "buildings_with_pv_positive": int(buildings_with_pv_positive),
            "buildings_with_pv_positive_share": (
                None if total_zone_buildings <= 0 else buildings_with_pv_positive / total_zone_buildings
            ),
        },
        "roof_availability": {
            "available_roof_area_total_m2": round(total_available_roof_area_m2, 2),
            "available_roof_area_by_orientation_m2": _rounded_orientation_area_dict(available_orientation_totals),
            "available_roof_share_by_orientation": {key: _round_or_none(available_orientation_share[key], 6) for key in ORIENTATION_ORDER},
            "pv_installed_area_by_orientation_m2": _rounded_orientation_area_dict(installed_orientation_totals),
            "pv_installed_share_by_orientation": {key: _round_or_none(installed_orientation_share[key], 6) for key in ORIENTATION_ORDER},
            "source": "panel_placement_summary" if bool(panel_availability_by_building) else "roof_surfaces_geojson",
        },
        "performance": {
            "kwh_per_installed_m2": _round_or_none(kwh_per_installed_m2, 4),
            "kwh_per_available_roof_m2": _round_or_none(kwh_per_available_roof_m2, 4),
            "roof_utilisation_ratio": _round_or_none(roof_utilisation_ratio, 6),
        },
        "quality_flags": {
            "invalid_or_ignored_roofs": int(roof_quality.get("ignored_roofs", 0)),
            "invalid_roofs_geometry": int(roof_quality.get("ignored_invalid_geometry", 0)),
            "invalid_roofs_zero_area": int(roof_quality.get("ignored_zero_area", 0)),
            "ignored_roofs_missing_building": int(roof_quality.get("ignored_missing_building", 0)),
            "valid_roofs": int(roof_quality.get("valid_roofs", 0)),
            "missing_pv_rows_for_zone_buildings": int(missing_pv_rows_for_zone),
            "pv_rows_not_in_zone": int(pv_rows_not_in_zone),
            "missing_pv_totals_file": not bool(pv_source_path),
            "missing_panel_code": pv_panel_efficiency is None,
            "missing_roof_availability_panel_summary": not bool(roof_availability_source_path),
        },
        "pv_panel": {
            "resolved_code": pv_panel_code,
            "efficiency": _round_or_none(pv_panel_efficiency, 6),
        },
    }
    return kpis, building_metrics


def render_html(output_path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Roof Visualizer</title>
  <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
  <script src="https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js"></script>
  <link href="https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.css" rel="stylesheet" />
  <style>
    html, body, #map {{ margin: 0; width: 100%; height: 100%; overflow: hidden; font-family: sans-serif; }}
    #panel {{
      position: absolute; top: 12px; left: 12px; z-index: 2; width: 380px; max-height: calc(100% - 24px);
      overflow: auto; padding: 12px; box-sizing: border-box; border-radius: 10px;
      color: #e5edf6; background: rgba(10, 18, 30, 0.88); border: 1px solid rgba(180, 200, 220, 0.25);
    }}
    h1 {{ margin: 0 0 8px; font-size: 18px; }}
    label {{ display: block; margin: 8px 0; }}
    .muted {{ color: #a9b8c8; font-size: 12px; overflow-wrap: anywhere; }}
    .metrics {{ white-space: pre-line; font-family: monospace; font-size: 12px; margin-top: 10px; }}
    #details {{ white-space: pre-line; font-family: monospace; font-size: 12px; margin-top: 10px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside id="panel">
    <h1>Roof Visualizer</h1>
    <div class="muted">Buildings: <span id="buildingCount"></span> | Roofs: <span id="roofCount"></span> | Bars: <span id="barCount"></span></div>
    <label><input id="zone2d" type="checkbox" checked> Zone footprints</label>
    <label><input id="zone3d" type="checkbox" checked> Zone 3D + roofs</label>
    <label><input id="zone3dOnly" type="checkbox"> Zone 3D only</label>
    <label><input id="solarBars" type="checkbox" checked> Solar bars</label>
    <div class="muted">Roof file: <span id="roofPath"></span></div>
    <div class="muted">PV file: <span id="pvPath"></span></div>
    <div class="muted">PV panel code: <span id="pvCode"></span> | Efficiency PV_n: <span id="pvEff"></span></div>
    <div id="kpiSummary" class="metrics"></div>
    <div id="details">Click a building, roof, or solar bar.</div>
  </aside>
  <script>
    const payload = {json.dumps(payload, ensure_ascii=True, separators=(",", ":"))};
    const state = {{zone2d: true, zone3d: true, zone3dOnly: false, solarBars: true}};
    let selectedBuilding = null;
    const details = document.getElementById("details");
    document.getElementById("buildingCount").textContent = String((payload.zone.features || []).length);
    document.getElementById("roofCount").textContent = String((payload.roofs || []).length);
    document.getElementById("barCount").textContent = String((payload.bars || []).length);
    document.getElementById("roofPath").textContent = payload.roof_file_path || "-";
    document.getElementById("pvPath").textContent = payload.pv_source_path || "(not found)";
    document.getElementById("pvCode").textContent = payload.pv_panel_code || "(not found)";
    document.getElementById("pvEff").textContent = payload.pv_panel_efficiency == null
      ? "(not found)"
      : Number(payload.pv_panel_efficiency).toLocaleString(undefined, {{maximumFractionDigits: 6}});

    function hasNumber(value) {{
      return value !== null && value !== undefined && Number.isFinite(Number(value));
    }}

    function n(value, digits = 2) {{
      if (!hasNumber(value)) return "n/a";
      return Number(value).toLocaleString(undefined, {{maximumFractionDigits: digits}});
    }}

    function pct(value, digits = 1) {{
      if (!hasNumber(value)) return "n/a";
      return `${{(Number(value) * 100).toLocaleString(undefined, {{maximumFractionDigits: digits}})}}%`;
    }}

    function normaliseBuildingName(value) {{
      const name = String(value || "").trim();
      return name || null;
    }}

    function isSelectedZoneFeature(feature) {{
      const featureName = normaliseBuildingName((feature && feature.properties && feature.properties.name) || "");
      return Boolean(featureName && selectedBuilding && featureName === selectedBuilding);
    }}

    function setSelectedBuilding(value) {{
      selectedBuilding = normaliseBuildingName(value);
      if (typeof deckgl !== "undefined" && deckgl) {{
        deckgl.setProps({{layers: layers()}});
      }}
    }}

    const orientationOrder = [
      ["flat", "Flat"],
      ["N", "North"],
      ["E", "East"],
      ["S", "South"],
      ["W", "West"]
    ];

    function formatOrientationLines(values, unit = "m2") {{
      return orientationOrder
        .map(([key, label]) => `${{label}}: ${{n((values || {{}})[key], 2)}} ${{unit}}`)
        .join("\\n");
    }}

    function formatKpiSummary() {{
      const k = payload.kpis || {{}};
      const totals = k.totals || {{}};
      const coverage = k.coverage || {{}};
      const roof = k.roof_availability || {{}};
      const quality = k.quality_flags || {{}};

      const lines = [
        "KPIs",
        "----",
        `PV area: ${{n(totals.pv_area_m2)}} m2`,
        `Installed power: ${{n(totals.installed_power_kwp, 3)}} kWp`,
        `PV generation: ${{n(totals.pv_generation_kwh)}} kWh/year (${{n(totals.pv_generation_mwh)}} MWh/year)`,
        "",
        "Coverage",
        "--------",
        `Custom roof buildings: ${{n(coverage.custom_roof_buildings, 0)}} / ${{n(coverage.total_zone_buildings, 0)}} (${{pct(coverage.custom_roof_share)}})`,
        `Buildings with PV > 0: ${{n(coverage.buildings_with_pv_positive, 0)}} (${{pct(coverage.buildings_with_pv_positive_share)}})`,
        "",
        "Roof Availability (custom roofs)",
        "-------------------------------",
        `Total available roof area: ${{n(roof.available_roof_area_total_m2)}} m2`,
        formatOrientationLines(roof.available_roof_area_by_orientation_m2, "m2"),
        "",
        "Data Quality",
        "------------",
        `Ignored or invalid roofs: ${{n(quality.invalid_or_ignored_roofs, 0)}}`,
        `Missing PV rows for zone buildings: ${{n(quality.missing_pv_rows_for_zone_buildings, 0)}}`,
        `Missing PV totals file: ${{quality.missing_pv_totals_file ? "yes" : "no"}}`,
        `Missing panel code: ${{quality.missing_panel_code ? "yes" : "no"}}`
      ];
      return lines.join("\\n");
    }}

    document.getElementById("kpiSummary").textContent = formatKpiSummary();

    function layers() {{
      const result = [];
      if (state.zone2d) {{
        result.push(new deck.GeoJsonLayer({{
          id: "zone-2d", data: payload.zone, pickable: true, stroked: true, filled: true,
          getFillColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 110] : [50, 130, 230, 55],
          getLineColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 255] : [30, 90, 200, 210],
          getLineWidth: f => isSelectedZoneFeature(f) ? 4 : 1,
          lineWidthMinPixels: 1
        }}));
      }}
      if (state.zone3d) {{
        result.push(new deck.GeoJsonLayer({{
          id: "zone-3d", data: payload.zone, pickable: true, stroked: true, filled: true,
          extruded: true, wireframe: true, opacity: 0.35,
          getFillColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 190] : [90, 170, 245, 115],
          getLineColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 255] : [245, 248, 255, 190],
          getLineWidth: f => isSelectedZoneFeature(f) ? 4 : 1,
          getElevation: f => Number((f.properties || {{}}).viz_height_m || 8)
        }}));
        result.push(new deck.PolygonLayer({{
          id: "roofs", data: payload.roofs || [], pickable: true, stroked: true, filled: true,
          getPolygon: d => d.polygon, getFillColor: d => d.fill_colour || [255, 140, 40, 220],
          getLineColor: [255, 255, 255, 170], lineWidthMinPixels: 1
        }}));
      }}
      if (state.zone3dOnly) {{
        result.push(new deck.GeoJsonLayer({{
          id: "zone-3d-only", data: payload.zone, pickable: true, stroked: true, filled: true,
          extruded: true, wireframe: true, opacity: 0.42,
          getFillColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 190] : [180, 190, 205, 120],
          getLineColor: f => isSelectedZoneFeature(f) ? [255, 215, 0, 255] : [240, 244, 250, 190],
          getLineWidth: f => isSelectedZoneFeature(f) ? 4 : 1,
          getElevation: f => Number((f.properties || {{}}).viz_height_m || 8)
        }}));
      }}
      if (state.solarBars) {{
        result.push(new deck.LineLayer({{
          id: "solar-bars", data: payload.bars || [], pickable: true, widthUnits: "meters",
          getWidth: d => Number(d.bar_width_m || {DEFAULT_BAR_WIDTH_M}),
          getSourcePosition: d => d.source_position, getTargetPosition: d => d.target_position,
          getColor: d => [...(d.bar_colour || [255, 140, 40]), 230]
        }}));
      }}
      return result;
    }}

    const buildingMetrics = payload.building_metrics || {{}};

    function dominantOrientationFromValues(values) {{
      let bestKey = null;
      let bestValue = 0;
      for (const [key] of orientationOrder) {{
        const value = Number((values || {{}})[key] || 0);
        if (Number.isFinite(value) && value > bestValue) {{
          bestValue = value;
          bestKey = key;
        }}
      }}
      return bestValue > 0 ? bestKey : null;
    }}

    function roofOrientationFromSummary(roof) {{
      const buildingName = normaliseBuildingName((roof && roof.building) || "");
      const bm = buildingName ? (buildingMetrics[buildingName] || null) : null;
      const summaryOrientation = dominantOrientationFromValues((bm && bm.available_by_orientation_m2) || null);
      if (summaryOrientation) return summaryOrientation;
      const geometryOrientation = String((roof && roof.orientation) || "").trim();
      return geometryOrientation || "-";
    }}

    function formatBuildingDetails(buildingName, fallbackHeight = null) {{
      const m = buildingMetrics[buildingName];
      if (!m) {{
        return [
          "Building",
          `Name: ${{buildingName || "-"}}`,
          `Height: ${{n(fallbackHeight)}} m`,
          "No metrics available for this building."
        ].join("\\n");
      }}

      const lines = [
        "Building",
        `Name: ${{buildingName}}`,
        `Height: ${{n(m.height_m)}} m`,
        `Custom roof detected: ${{m.custom_roof_detected ? "yes" : "no"}}`,
        `Custom roofs: ${{n(m.custom_roof_count, 0)}}`,
        `Available roof area: ${{n(m.available_roof_area_m2)}} m2`,
        "PV installed area by orientation:",
        formatOrientationLines(m.pv_installed_by_orientation_m2, "m2"),
        `PV generation: ${{n(m.pv_generation_kwh)}} kWh/year`,
        `PV installed area: ${{n(m.pv_installed_area_m2)}} m2`,
        `Installed power: ${{n(m.pv_installed_power_kwp, 3)}} kWp`
      ];
      return lines.join("\\n");
    }}

    const deckgl = new deck.DeckGL({{
      container: "map", mapStyle: payload.map_style, controller: true, layers: layers(),
      initialViewState: {{latitude: payload.centre_lat, longitude: payload.centre_lon, zoom: 13, pitch: 55, bearing: -15}},
      getTooltip: info => {{
        if (!info.object) return null;
        if (info.layer.id === "roofs") {{
          return `Roof\\nBuilding: ${{info.object.building}}\\nRoof: ${{info.object.roof_id}}\\nArea: ${{n(info.object.area_m2)}} m2\\nOrientation: ${{roofOrientationFromSummary(info.object)}}`;
        }}
        if (info.layer.id === "solar-bars") {{
          const bm = buildingMetrics[info.object.building] || {{}};
          return `Solar\\nBuilding: ${{info.object.building}}\\n${{n(info.object.solar_kwh, 1)}} kWh/year\\nInstalled: ${{n(bm.pv_installed_power_kwp, 3)}} kWp`;
        }}
        const p = info.object.properties || {{}};
        const bm = buildingMetrics[p.name || ""] || {{}};
        return `Building\\nName: ${{p.name || "-"}}\\nHeight: ${{n(p.viz_height_m)}} m\\nAvailable roof: ${{n(bm.available_roof_area_m2)}} m2`;
      }},
      onClick: info => {{
        if (!info.object) return;
        if (info.layer.id === "roofs") {{
          const buildingName = info.object.building || "-";
          setSelectedBuilding(buildingName);
          details.textContent = formatBuildingDetails(buildingName);
          return;
        }} else if (info.layer.id === "solar-bars") {{
          setSelectedBuilding(info.object.building);
          return;
        }} else {{
          const p = info.object.properties || {{}};
          const buildingName = p.name || "-";
          setSelectedBuilding(buildingName);
          details.textContent = formatBuildingDetails(buildingName, p.viz_height_m);
        }}
      }}
    }});

    for (const id of Object.keys(state)) {{
      document.getElementById(id).addEventListener("change", event => {{
        state[id] = Boolean(event.target.checked);
        deckgl.setProps({{layers: layers()}});
      }});
    }}
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
    if args.coordinate_precision < 0:
        raise ValueError("--coordinate-precision must be non-negative.")
    if args.flat_threshold_deg < 0:
        raise ValueError("--flat-threshold-deg must be non-negative.")

    scenario = os.path.abspath(args.scenario)
    if not os.path.isdir(scenario):
        raise NotADirectoryError(f"Scenario folder does not exist: {scenario}")

    locator = InputLocator(scenario)
    roof_file_path = _resolve_path(_default_roof_file_path(scenario), args.roof_file)
    roofs, roof_stats, roof_quality = _load_roofs(
        roof_file_path=roof_file_path,
        precision=args.coordinate_precision,
        flat_threshold_deg=float(args.flat_threshold_deg),
    )
    zone, zone_centroids, centre, bounds, zone_heights_by_building, zone_buildings = _load_zone_payload(
        locator=locator,
        roof_stats=roof_stats,
        precision=args.coordinate_precision,
    )

    pv_metrics_by_building, pv_source_path = _load_pv_metrics_by_building(locator, args.pv_panel)
    pv_panel_efficiency, pv_panel_code = _load_pv_panel_efficiency(locator, args.pv_panel)
    roof_availability_by_building, _, roof_availability_source_path = _load_panel_placement_orientation_by_building(
        locator, args.pv_panel
    )

    solar_by_building = {
        building: float(metrics.get("E_PV_gen_kWh", 0.0))
        for building, metrics in pv_metrics_by_building.items()
    }
    bars = _build_solar_bars(
        roof_stats=roof_stats,
        zone_centroids=zone_centroids,
        zone_heights_by_building=zone_heights_by_building,
        solar_by_building_kwh=solar_by_building,
        bar_scale_m_per_mwh=float(args.bar_scale_m_per_mwh),
        min_bar_height_m=float(args.min_bar_height_m),
        bar_width_m=float(args.bar_width_m),
        precision=args.coordinate_precision,
    )
    kpis, building_metrics = _build_decision_metrics(
        zone_buildings=zone_buildings,
        zone_heights_by_building=zone_heights_by_building,
        roof_stats=roof_stats,
        roof_quality=roof_quality,
        pv_metrics_by_building=pv_metrics_by_building,
        pv_source_path=pv_source_path,
        pv_panel_efficiency=pv_panel_efficiency,
        pv_panel_code=pv_panel_code,
        roof_availability_by_building=roof_availability_by_building,
        roof_availability_source_path=roof_availability_source_path,
    )

    payload = {
        "scenario": scenario,
        "pv_panel": str(args.pv_panel),
        "pv_panel_code": pv_panel_code,
        "pv_panel_efficiency": pv_panel_efficiency,
        "flat_threshold_deg": float(args.flat_threshold_deg),
        "map_style": args.map_style,
        "zone": zone,
        "roofs": roofs,
        "bars": bars,
        "kpis": kpis,
        "building_metrics": building_metrics,
        "roof_quality": roof_quality,
        "centre_lat": centre[0],
        "centre_lon": centre[1],
        "bounds": {"west": bounds[0], "south": bounds[1], "east": bounds[2], "north": bounds[3]},
        "roof_file_path": roof_file_path,
        "pv_source_path": pv_source_path,
        "roof_availability_source_path": roof_availability_source_path,
    }

    output_path = _resolve_path(_default_output_html_path(scenario), args.output_html)
    render_html(output_path, payload)
    print("[ok] Roof visualizer generated.")
    print(f"Buildings loaded: {len(zone.get('features', []))}")
    print(f"Roofs loaded: {len(roofs)}")
    print(f"Solar bars loaded: {len(bars)}")
    print(f"PV source: {pv_source_path or '(not found)'}")
    print(f"Roof availability source: {roof_availability_source_path or '(not found)'}")
    print(f"PV panel efficiency (PV_n): {pv_panel_efficiency if pv_panel_efficiency is not None else '(not found)'}")
    print(f"Output HTML: {output_path}")


if __name__ == "__main__":
    main()
