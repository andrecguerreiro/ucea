"""Build per-area Excel comparison files between validation and UCEA radiation outputs.

The validation CSVs are the source of truth for which buildings to compare
(residential filter is already applied there).

Inputs (defaults):
- Validation outputs root: tools/validation_outputs
- Areas config: tools/validation_areas_config.json
- UCEA scenarios root: C:/Users/Andre/cea-scenarios/Validation

Outputs:
- One workbook per area: <area_id>_radiation_comparison.xlsx
- One aggregate workbook: all_areas_radiation_comparison.xlsx
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
import xml.sax.saxutils as saxutils
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UCEA_ROOFTOP_SUMMARY_RELATIVE = Path("outputs/data/solar-radiation/building_rooftop_radiation_annual_kWh.csv")
WORKFLOW_COMPARISON_RELATIVE = Path("outputs/data/roof-workflow-comparison")
WORKFLOW_0_NAME = "workflow0_normal_flat_roofs"
WORKFLOW_1_NAME = "workflow1_geometry_generator"
WORKFLOW_0_SOLAR_RELATIVE = WORKFLOW_COMPARISON_RELATIVE / WORKFLOW_0_NAME / "solar-radiation"
WORKFLOW_1_SOLAR_RELATIVE = WORKFLOW_COMPARISON_RELATIVE / WORKFLOW_1_NAME / "solar-radiation"


@dataclass
class AreaComparison:
    area_id: str
    validation_csv_path: Path
    ucea_summary_csv_path: Path | None
    comparison_rows: list[dict[str, Any]]
    validation_missing_in_ucea_rows: list[dict[str, Any]]
    ucea_only_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def _to_absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sanitize_sheet_name(name: str) -> str:
    invalid = r"[]:*?/\\"
    sanitized = "".join("_" if ch in invalid else ch for ch in name).strip()
    sanitized = sanitized or "Sheet"
    return sanitized[:31]


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", name)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _discover_validation_csvs(validation_root: Path) -> list[Path]:
    return sorted(validation_root.rglob("*_residential_radiation.csv"))


def _deduplicate_validation_csvs_by_area(validation_csvs: list[Path]) -> list[Path]:
    """
    Keep one validation CSV per area id.
    If duplicates exist, pick the most recently modified file and print a warning.
    """
    grouped: dict[str, list[Path]] = {}
    for path in validation_csvs:
        area_id = path.stem.replace("_residential_radiation", "").strip()
        grouped.setdefault(area_id, []).append(path)

    selected: list[Path] = []
    for area_id in sorted(grouped):
        candidates = grouped[area_id]
        if len(candidates) == 1:
            selected.append(candidates[0])
            continue

        ranked = sorted(
            candidates,
            key=lambda p: (p.stat().st_mtime, len(str(p))),
            reverse=True,
        )
        winner = ranked[0]
        dropped = ranked[1:]
        print(
            f"[warn] Multiple validation CSVs found for area '{area_id}'. "
            f"Using newest: {winner}"
        )
        for dup in dropped:
            print(f"[warn]   Skipping duplicate: {dup}")
        selected.append(winner)

    return selected


def _load_removed_zero_radiation_buildings(validation_csv_path: Path, area_id: str) -> list[str]:
    summary_path = validation_csv_path.parent / f"{area_id}_validation_summary.json"
    if not summary_path.exists():
        return []
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = payload.get("removed_zero_radiation_buildings", [])
    if not isinstance(values, list):
        return []
    result = [str(v).strip() for v in values if str(v).strip()]
    return sorted(set(result))


def _load_area_to_scenario_from_config(config_path: Path) -> dict[str, Path]:
    if not config_path.exists():
        return {}

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}

    mapping: dict[str, Path] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        area_id = str(row.get("area_id") or "").strip()
        zone_shp_path = str(row.get("zone_shp_path") or "").strip()
        if not area_id or not zone_shp_path:
            continue

        zone_path = Path(zone_shp_path)
        # .../<scenario>/inputs/building-geometry/zone.shp -> <scenario>
        parts = [p.lower() for p in zone_path.parts]
        if "inputs" in parts:
            idx = parts.index("inputs")
            if idx - 1 >= 0:
                scenario_dir = Path(*zone_path.parts[:idx])
                mapping[area_id] = _to_absolute(scenario_dir)
    return mapping


def _match_scenario_dir(
    area_id: str,
    validation_area_dirname: str,
    area_to_scenario_from_config: dict[str, Path],
    ucea_root: Path,
) -> Path | None:
    # 1) explicit config mapping
    config_hit = area_to_scenario_from_config.get(area_id)
    if config_hit and (config_hit / UCEA_ROOFTOP_SUMMARY_RELATIVE).exists():
        return config_hit

    # 2) direct folder name from validation area directory
    direct = ucea_root / validation_area_dirname
    if (direct / UCEA_ROOFTOP_SUMMARY_RELATIVE).exists():
        return direct

    # 3) normalized-name matching fallback
    norm_targets = {_normalize_key(area_id), _normalize_key(validation_area_dirname)}
    for child in sorted([p for p in ucea_root.iterdir() if p.is_dir()]):
        child_norm = _normalize_key(child.name)
        if child_norm in norm_targets and (child / UCEA_ROOFTOP_SUMMARY_RELATIVE).exists():
            return child

    # 4) token-subset fallback (handles light naming mismatches)
    target_tokens = set(_normalize_key(area_id).split()) | set(_normalize_key(validation_area_dirname).split())
    for child in sorted([p for p in ucea_root.iterdir() if p.is_dir()]):
        child_tokens = set(_normalize_key(child.name).split())
        if child_tokens and child_tokens.issubset(target_tokens):
            if (child / UCEA_ROOFTOP_SUMMARY_RELATIVE).exists():
                return child

    return None


def _aggregate_validation_buildings(validation_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in validation_rows:
        building = str(row.get("cea_name") or "").strip()
        if not building:
            continue

        area_m2 = _to_float_or_none(row.get("intersection_area_m2"))
        radiation_kwh = _to_float_or_none(row.get("radiation_sum_kwh_year"))
        if radiation_kwh is None:
            radiation_kwh = _to_float_or_none(row.get("radiation_sum"))

        point_count = _to_float_or_none(row.get("radiation_point_count"))

        if building not in result:
            result[building] = {
                "validation_area_m2": 0.0,
                "validation_radiation_kwh_year": 0.0,
                "validation_point_count": 0.0,
                "residential_source": str(row.get("residential_source") or "").strip(),
                "use_type1": str(row.get("use_type1") or "").strip(),
                "use_type2": str(row.get("use_type2") or "").strip(),
                "use_type3": str(row.get("use_type3") or "").strip(),
                "resi_type": str(row.get("resi_type") or "").strip(),
            }

        if area_m2 is not None:
            result[building]["validation_area_m2"] += area_m2
        if radiation_kwh is not None:
            result[building]["validation_radiation_kwh_year"] += radiation_kwh
        if point_count is not None:
            result[building]["validation_point_count"] += point_count

    # Clean zero-like aggregates to None where appropriate
    for item in result.values():
        if item["validation_point_count"] == 0.0:
            item["validation_point_count"] = None
    return result


def _read_ucea_summary_buildings(ucea_summary_csv_path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_csv_rows(ucea_summary_csv_path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        building = str(row.get("name") or "").strip()
        if not building:
            continue
        result[building] = {
            "ucea_roofs_top_m2": _to_float_or_none(row.get("roofs_top_m2")),
            "ucea_rooftop_radiation_kwh_year": _to_float_or_none(row.get("rooftop_radiation_kWh_year")),
            "ucea_timesteps": _to_float_or_none(row.get("timesteps")),
            "ucea_source_column": str(row.get("source_column") or "").strip(),
            "ucea_oven_confidence_face_count": _to_float_or_none(row.get("oven_confidence_face_count")),
            "ucea_oven_confidence_mean": _to_float_or_none(row.get("oven_confidence_mean")),
            "ucea_oven_confidence_area_weighted_mean": _to_float_or_none(
                row.get("oven_confidence_area_weighted_mean")
            ),
            "ucea_oven_confidence_min": _to_float_or_none(row.get("oven_confidence_min")),
            "ucea_oven_confidence_max": _to_float_or_none(row.get("oven_confidence_max")),
        }
    return result


def _read_rooftop_summary_from_radiation_csv(path: Path) -> tuple[float, float | None, int, str] | None:
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return None

        source_column = ""
        if "roofs_top_kW" in reader.fieldnames:
            source_column = "roofs_top_kW"
        elif "roofs_top_kWh" in reader.fieldnames:
            source_column = "roofs_top_kWh"
        else:
            return None

        annual_rooftop_radiation_kwh = 0.0
        roof_area_m2: float | None = None
        timesteps = 0

        for row in reader:
            timesteps += 1
            radiation_value = _to_float_or_none(row.get(source_column))
            if radiation_value is not None:
                annual_rooftop_radiation_kwh += radiation_value
            if roof_area_m2 is None:
                roof_area_value = _to_float_or_none(row.get("roofs_top_m2"))
                if roof_area_value is not None:
                    roof_area_m2 = roof_area_value
    return annual_rooftop_radiation_kwh, roof_area_m2, timesteps, source_column


def _read_workflow_summary_buildings(
    workflow_solar_radiation_dir: Path, workflow_prefix: str
) -> dict[str, dict[str, Any]]:
    if not workflow_solar_radiation_dir.exists():
        return {}

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(workflow_solar_radiation_dir.glob("*_radiation.csv")):
        building = path.stem.replace("_radiation", "").strip()
        if not building:
            continue
        summary = _read_rooftop_summary_from_radiation_csv(path)
        if summary is None:
            continue
        annual_rooftop_radiation_kwh, roof_area_m2, timesteps, source_column = summary
        result[building] = {
            f"{workflow_prefix}_roofs_top_m2": roof_area_m2,
            f"{workflow_prefix}_rooftop_radiation_kwh_year": annual_rooftop_radiation_kwh,
            f"{workflow_prefix}_timesteps": timesteps,
            f"{workflow_prefix}_source_column": source_column,
        }
    return result


def _read_workflow0_summary_buildings(workflow0_solar_radiation_dir: Path) -> dict[str, dict[str, Any]]:
    return _read_workflow_summary_buildings(workflow0_solar_radiation_dir, "workflow0")


def _read_workflow1_summary_buildings(workflow1_solar_radiation_dir: Path) -> dict[str, dict[str, Any]]:
    return _read_workflow_summary_buildings(workflow1_solar_radiation_dir, "workflow1")


def _read_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_geojson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def _safe_subtract(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return None
    return lhs - rhs


def _find_validation_residential_geojson(validation_csv_path: Path, area_id: str) -> Path | None:
    candidate = validation_csv_path.parent / f"{area_id}_residential_buildings.geojson"
    if candidate.exists():
        return candidate
    return None


def _augment_feature_for_error_map(feature: dict[str, Any], comparison_row: dict[str, Any]) -> dict[str, Any]:
    properties = dict(feature.get("properties", {}))
    validation_radiation = _to_float_or_none(comparison_row.get("validation_radiation_kwh_year"))
    ucea_radiation = _to_float_or_none(comparison_row.get("ucea_radiation_kwh_year"))
    workflow0_radiation = _to_float_or_none(comparison_row.get("workflow0_radiation_kwh_year"))
    validation_area = _to_float_or_none(comparison_row.get("validation_area_m2"))
    ucea_area = _to_float_or_none(comparison_row.get("ucea_roofs_top_m2"))

    ucea_minus_validation_radiation = (
        (ucea_radiation - validation_radiation)
        if (ucea_radiation is not None and validation_radiation is not None)
        else None
    )
    workflow0_minus_validation_radiation = (
        (workflow0_radiation - validation_radiation)
        if (workflow0_radiation is not None and validation_radiation is not None)
        else None
    )
    ucea_minus_validation_area = (
        (ucea_area - validation_area) if (ucea_area is not None and validation_area is not None) else None
    )

    ucea_error_abs = abs(ucea_minus_validation_radiation) if ucea_minus_validation_radiation is not None else None
    workflow0_error_abs = (
        abs(workflow0_minus_validation_radiation) if workflow0_minus_validation_radiation is not None else None
    )
    ucea_error_pct_raw = _safe_divide(ucea_minus_validation_radiation, validation_radiation)
    workflow0_error_pct_raw = _safe_divide(workflow0_minus_validation_radiation, validation_radiation)
    ucea_area_error_pct_raw = _safe_divide(ucea_minus_validation_area, validation_area)

    ucea_error_pct = abs(ucea_error_pct_raw) * 100.0 if ucea_error_pct_raw is not None else None
    workflow0_error_pct = abs(workflow0_error_pct_raw) * 100.0 if workflow0_error_pct_raw is not None else None
    ucea_area_error_pct = abs(ucea_area_error_pct_raw) * 100.0 if ucea_area_error_pct_raw is not None else None

    properties.update(
        {
            "comparison_status": comparison_row.get("status"),
            "validation_radiation_kwh_year": validation_radiation,
            "ucea_radiation_kwh_year": ucea_radiation,
            "workflow0_radiation_kwh_year": workflow0_radiation,
            "validation_area_m2": validation_area,
            "ucea_roofs_top_m2": ucea_area,
            "err_ucea_minus_validation_kwh_year": ucea_error_abs,
            "err_ucea_abs_kwh_year": ucea_error_abs,
            "err_ucea_pct_of_validation": ucea_error_pct,
            "err_ucea_area_minus_validation_m2": abs(ucea_minus_validation_area)
            if ucea_minus_validation_area is not None
            else None,
            "err_ucea_area_pct_of_validation": ucea_area_error_pct,
            "err_workflow0_minus_validation_kwh_year": workflow0_error_abs,
            "err_workflow0_abs_kwh_year": workflow0_error_abs,
            "err_workflow0_pct_of_validation": workflow0_error_pct,
        }
    )
    return {
        "type": feature.get("type", "Feature"),
        "properties": properties,
        "geometry": feature.get("geometry"),
    }


def _write_area_error_map_geojson(
    output_path: Path,
    validation_residential_geojson_path: Path | None,
    area_comparison: AreaComparison,
) -> None:
    if validation_residential_geojson_path is None or not validation_residential_geojson_path.exists():
        return

    payload = _read_geojson(validation_residential_geojson_path)
    if payload.get("type") != "FeatureCollection":
        return

    comparison_by_building = {
        str(row.get("building_name") or "").strip(): row for row in area_comparison.comparison_rows
    }
    features = payload.get("features", [])
    out_features: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        building_name = str((props or {}).get("cea_name") or "").strip()
        comparison_row = comparison_by_building.get(building_name)
        if comparison_row is None:
            continue
        out_features.append(_augment_feature_for_error_map(feature, comparison_row))

    out_payload = {
        "type": "FeatureCollection",
        "name": f"{area_comparison.area_id}_radiation_error_map",
        "features": out_features,
    }
    _write_geojson(output_path, out_payload)


def _write_combined_error_map_geojson(output_path: Path, area_error_map_paths: list[Path]) -> None:
    out_features: list[dict[str, Any]] = []
    for area_path in area_error_map_paths:
        if not area_path.exists():
            continue
        payload = _read_geojson(area_path)
        if payload.get("type") != "FeatureCollection":
            continue
        source_name = area_path.stem
        for feature in payload.get("features", []):
            if not isinstance(feature, dict):
                continue
            props = dict(feature.get("properties", {}))
            props["source_error_map"] = source_name
            out_features.append(
                {
                    "type": feature.get("type", "Feature"),
                    "properties": props,
                    "geometry": feature.get("geometry"),
                }
            )

    out_payload = {
        "type": "FeatureCollection",
        "name": "all_areas_radiation_error_map",
        "features": out_features,
    }
    _write_geojson(output_path, out_payload)


def _build_area_comparison(
    area_id: str,
    validation_csv_path: Path,
    ucea_summary_csv_path: Path | None,
    workflow0_buildings: dict[str, dict[str, Any]] | None = None,
    workflow1_buildings: dict[str, dict[str, Any]] | None = None,
) -> AreaComparison:
    validation_rows = _read_csv_rows(validation_csv_path)
    validation_buildings = _aggregate_validation_buildings(validation_rows)
    ucea_buildings = _read_ucea_summary_buildings(ucea_summary_csv_path) if ucea_summary_csv_path else {}
    removed_zero_radiation_buildings = _load_removed_zero_radiation_buildings(validation_csv_path, area_id)

    workflow0_buildings = workflow0_buildings or {}
    workflow1_buildings = workflow1_buildings or {}

    comparison_rows: list[dict[str, Any]] = []
    validation_missing_in_ucea_rows: list[dict[str, Any]] = []

    for building in sorted(validation_buildings):
        v = validation_buildings[building]
        u = ucea_buildings.get(building, {})
        wf0 = workflow0_buildings.get(building, {})
        wf1 = workflow1_buildings.get(building, {})

        validation_area_m2 = v.get("validation_area_m2")
        ucea_area_m2 = wf1.get("workflow1_roofs_top_m2")
        if ucea_area_m2 is None:
            ucea_area_m2 = u.get("ucea_roofs_top_m2")
        validation_radiation_kwh = v.get("validation_radiation_kwh_year")
        ucea_radiation_kwh = wf1.get("workflow1_rooftop_radiation_kwh_year")
        if ucea_radiation_kwh is None:
            ucea_radiation_kwh = u.get("ucea_rooftop_radiation_kwh_year")
        ucea_source_column = wf1.get("workflow1_source_column")
        if not ucea_source_column:
            ucea_source_column = u.get("ucea_source_column")
        ucea_timesteps = wf1.get("workflow1_timesteps")
        if ucea_timesteps is None:
            ucea_timesteps = u.get("ucea_timesteps")
        workflow0_radiation_kwh = wf0.get("workflow0_rooftop_radiation_kwh_year")
        error_cea = _safe_subtract(workflow0_radiation_kwh, validation_radiation_kwh)
        error_cea_oven = _safe_subtract(ucea_radiation_kwh, validation_radiation_kwh)

        row = {
            "area_id": area_id,
            "building_name": building,
            "status": "matched" if u else "missing_in_ucea",
            "validation_area_m2": validation_area_m2,
            "ucea_roofs_top_m2": ucea_area_m2,
            "validation_radiation_kwh_year": validation_radiation_kwh,
            "ucea_radiation_kwh_year": ucea_radiation_kwh,
            "workflow0_roofs_top_m2": wf0.get("workflow0_roofs_top_m2"),
            "workflow0_radiation_kwh_year": workflow0_radiation_kwh,
            "error_cea": error_cea,
            "error_cea_oven": error_cea_oven,
            "workflow0_source_column": wf0.get("workflow0_source_column"),
            "workflow0_timesteps": wf0.get("workflow0_timesteps"),
            "validation_point_count": v.get("validation_point_count"),
            "residential_source": v.get("residential_source"),
            "use_type1": v.get("use_type1"),
            "use_type2": v.get("use_type2"),
            "use_type3": v.get("use_type3"),
            "resi_type": v.get("resi_type"),
            "ucea_source_column": ucea_source_column,
            "ucea_timesteps": ucea_timesteps,
            "ucea_oven_confidence_face_count": u.get("ucea_oven_confidence_face_count"),
            "ucea_oven_confidence_mean": u.get("ucea_oven_confidence_mean"),
            "ucea_oven_confidence_area_weighted_mean": u.get("ucea_oven_confidence_area_weighted_mean"),
            "ucea_oven_confidence_min": u.get("ucea_oven_confidence_min"),
            "ucea_oven_confidence_max": u.get("ucea_oven_confidence_max"),
        }
        comparison_rows.append(row)
        if not u:
            validation_missing_in_ucea_rows.append(row)

    ucea_only_rows: list[dict[str, Any]] = []
    for building in sorted(set(ucea_buildings) - set(validation_buildings)):
        u = ucea_buildings[building]
        wf0 = workflow0_buildings.get(building, {})
        wf1 = workflow1_buildings.get(building, {})
        ucea_area_m2 = wf1.get("workflow1_roofs_top_m2")
        if ucea_area_m2 is None:
            ucea_area_m2 = u.get("ucea_roofs_top_m2")
        ucea_radiation_kwh = wf1.get("workflow1_rooftop_radiation_kwh_year")
        if ucea_radiation_kwh is None:
            ucea_radiation_kwh = u.get("ucea_rooftop_radiation_kwh_year")
        ucea_source_column = wf1.get("workflow1_source_column")
        if not ucea_source_column:
            ucea_source_column = u.get("ucea_source_column")
        ucea_timesteps = wf1.get("workflow1_timesteps")
        if ucea_timesteps is None:
            ucea_timesteps = u.get("ucea_timesteps")
        ucea_only_rows.append(
            {
                "area_id": area_id,
                "building_name": building,
                "ucea_roofs_top_m2": ucea_area_m2,
                "ucea_radiation_kwh_year": ucea_radiation_kwh,
                "workflow0_roofs_top_m2": wf0.get("workflow0_roofs_top_m2"),
                "workflow0_radiation_kwh_year": wf0.get("workflow0_rooftop_radiation_kwh_year"),
                "error_cea": None,
                "error_cea_oven": None,
                "workflow0_source_column": wf0.get("workflow0_source_column"),
                "workflow0_timesteps": wf0.get("workflow0_timesteps"),
                "ucea_source_column": ucea_source_column,
                "ucea_timesteps": ucea_timesteps,
                "ucea_oven_confidence_face_count": u.get("ucea_oven_confidence_face_count"),
                "ucea_oven_confidence_mean": u.get("ucea_oven_confidence_mean"),
                "ucea_oven_confidence_area_weighted_mean": u.get("ucea_oven_confidence_area_weighted_mean"),
                "ucea_oven_confidence_min": u.get("ucea_oven_confidence_min"),
                "ucea_oven_confidence_max": u.get("ucea_oven_confidence_max"),
            }
        )

    def _sum_key(rows: list[dict[str, Any]], key: str) -> float:
        total = 0.0
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                total += float(value)
        return total

    matched_rows = [r for r in comparison_rows if r["status"] == "matched"]
    total_validation_buildings_with_custom_roofs = sum(
        1
        for row in comparison_rows
        if isinstance(row.get("ucea_oven_confidence_face_count"), (int, float))
        and math.isfinite(float(row["ucea_oven_confidence_face_count"]))
        and float(row["ucea_oven_confidence_face_count"]) > 0.0
    )
    total_validation_area_m2 = _sum_key(comparison_rows, "validation_area_m2")
    total_ucea_area_m2 = _sum_key(matched_rows, "ucea_roofs_top_m2")
    total_workflow0_area_m2 = _sum_key(matched_rows, "workflow0_roofs_top_m2")
    total_validation_radiation_kwh = _sum_key(comparison_rows, "validation_radiation_kwh_year")
    total_ucea_radiation_kwh = _sum_key(matched_rows, "ucea_radiation_kwh_year")
    total_workflow0_radiation_kwh = _sum_key(matched_rows, "workflow0_radiation_kwh_year")

    summary_rows = [
        {"metric": "area_id", "value": area_id},
        {"metric": "validation_csv_path", "value": str(validation_csv_path)},
        {"metric": "ucea_summary_csv_path", "value": str(ucea_summary_csv_path) if ucea_summary_csv_path else ""},
        {"metric": "validation_buildings_total", "value": len(comparison_rows)},
        {
            "metric": "total_validation_buildings_with_custom_roofs",
            "value": total_validation_buildings_with_custom_roofs,
        },
        {
            "metric": "removed_zero_radiation_buildings_count",
            "value": len(removed_zero_radiation_buildings),
        },
        {
            "metric": "removed_zero_radiation_buildings",
            "value": ", ".join(removed_zero_radiation_buildings),
        },
        {"metric": "matched_buildings", "value": len(matched_rows)},
        {"metric": "validation_missing_in_ucea", "value": len(validation_missing_in_ucea_rows)},
        {"metric": "ucea_only_buildings", "value": len(ucea_only_rows)},
        {"metric": "total_validation_area_m2", "value": total_validation_area_m2},
        {"metric": "total_ucea_roofs_top_m2_matched", "value": total_ucea_area_m2},
        {"metric": "total_workflow0_roofs_top_m2_matched", "value": total_workflow0_area_m2},
        {"metric": "total_validation_radiation_kwh_year", "value": total_validation_radiation_kwh},
        {"metric": "total_ucea_radiation_kwh_year_matched", "value": total_ucea_radiation_kwh},
        {"metric": "total_workflow0_radiation_kwh_year_matched", "value": total_workflow0_radiation_kwh},
    ]

    return AreaComparison(
        area_id=area_id,
        validation_csv_path=validation_csv_path,
        ucea_summary_csv_path=ucea_summary_csv_path,
        comparison_rows=comparison_rows,
        validation_missing_in_ucea_rows=validation_missing_in_ucea_rows,
        ucea_only_rows=ucea_only_rows,
        summary_rows=summary_rows,
    )


def _to_worksheet_table(rows: list[dict[str, Any]], preferred_headers: list[str] | None = None) -> list[list[Any]]:
    if preferred_headers:
        headers = preferred_headers
    elif rows:
        headers = list(rows[0].keys())
    else:
        headers = []
    table = [headers]
    for row in rows:
        table.append([row.get(col) for col in headers])
    return table


def _col_to_a1(col_idx_zero_based: int) -> str:
    col = col_idx_zero_based + 1
    chars: list[str] = []
    while col:
        col, rem = divmod(col - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def _xml_escape_text(value: str) -> str:
    return saxutils.escape(value, {"'": "&apos;", '"': "&quot;"})


def _value_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _build_sheet_xml(table: list[list[Any]]) -> str:
    rows_xml: list[str] = []
    for r_idx, row in enumerate(table, start=1):
        cells_xml: list[str] = []
        for c_idx, value in enumerate(row):
            if value is None or value == "":
                continue
            cell_ref = f"{_col_to_a1(c_idx)}{r_idx}"
            if _value_is_number(value):
                cells_xml.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = _xml_escape_text(str(value))
                cells_xml.append(f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        rows_xml.append(f'<row r="{r_idx}">{"".join(cells_xml)}</row>')

    last_col = _col_to_a1(max((len(r) for r in table), default=1) - 1)
    last_row = max(len(table), 1)
    dimension = f"A1:{last_col}{last_row}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f'<sheetData>{"".join(rows_xml)}</sheetData>'
        "</worksheet>"
    )


def _write_xlsx_minimal(path: Path, sheets: list[tuple[str, list[list[Any]]]]) -> None:
    safe_sheets: list[tuple[str, list[list[Any]]]] = []
    used_names: set[str] = set()
    for raw_name, table in sheets:
        name = _sanitize_sheet_name(raw_name)
        if name in used_names:
            suffix = 2
            while f"{name[:28]}_{suffix}" in used_names:
                suffix += 1
            name = f"{name[:28]}_{suffix}"
        used_names.add(name)
        safe_sheets.append((name, table))

    workbook_sheets_xml = []
    workbook_rels_xml = []
    content_types_overrides = []
    worksheet_xmls: list[str] = []

    for idx, (name, table) in enumerate(safe_sheets, start=1):
        worksheet_xmls.append(_build_sheet_xml(table))
        workbook_sheets_xml.append(
            f'<sheet name="{_xml_escape_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels_xml.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        content_types_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f'{"".join(content_types_overrides)}'
        "</Types>"
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets_xml)}</sheets>'
        "</workbook>"
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels_xml)}'
        '<Relationship Id="rIdStyles" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
        "</Relationships>"
    )

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
        '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
        '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
        "</styleSheet>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/styles.xml", styles_xml)
        for idx, worksheet_xml in enumerate(worksheet_xmls, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml)


def _write_area_workbook(output_path: Path, area_comparison: AreaComparison) -> None:
    summary_table = _to_worksheet_table(area_comparison.summary_rows, ["metric", "value"])
    comparison_headers = [
        "area_id",
        "building_name",
        "status",
        "validation_area_m2",
        "ucea_roofs_top_m2",
        "workflow0_roofs_top_m2",
        "validation_radiation_kwh_year",
        "ucea_radiation_kwh_year",
        "workflow0_radiation_kwh_year",
        "error_cea",
        "error_cea_oven",
        "workflow0_source_column",
        "workflow0_timesteps",
        "validation_point_count",
        "residential_source",
        "use_type1",
        "use_type2",
        "use_type3",
        "resi_type",
        "ucea_source_column",
        "ucea_timesteps",
        "ucea_oven_confidence_face_count",
        "ucea_oven_confidence_mean",
        "ucea_oven_confidence_area_weighted_mean",
        "ucea_oven_confidence_min",
        "ucea_oven_confidence_max",
    ]
    comparison_table = _to_worksheet_table(area_comparison.comparison_rows, comparison_headers)
    missing_table = _to_worksheet_table(area_comparison.validation_missing_in_ucea_rows, comparison_headers)
    ucea_only_table = _to_worksheet_table(
        area_comparison.ucea_only_rows,
        [
            "area_id",
            "building_name",
            "ucea_roofs_top_m2",
            "ucea_radiation_kwh_year",
            "workflow0_roofs_top_m2",
            "workflow0_radiation_kwh_year",
            "error_cea",
            "error_cea_oven",
            "workflow0_source_column",
            "workflow0_timesteps",
            "ucea_source_column",
            "ucea_timesteps",
            "ucea_oven_confidence_face_count",
            "ucea_oven_confidence_mean",
            "ucea_oven_confidence_area_weighted_mean",
            "ucea_oven_confidence_min",
            "ucea_oven_confidence_max",
        ],
    )

    sheets = [
        ("summary", summary_table),
        ("comparison", comparison_table),
        ("missing_in_ucea", missing_table),
        ("ucea_only", ucea_only_table),
    ]
    _write_xlsx_minimal(output_path, sheets)


def _write_overall_workbook(output_path: Path, area_comparisons: list[AreaComparison]) -> None:
    overall_summary_rows: list[dict[str, Any]] = []
    all_comparison_rows: list[dict[str, Any]] = []
    for area in area_comparisons:
        summary_dict = {row["metric"]: row["value"] for row in area.summary_rows}
        overall_summary_rows.append(
            {
                "area_id": area.area_id,
                "validation_buildings_total": summary_dict.get("validation_buildings_total"),
                "total_validation_buildings_with_custom_roofs": summary_dict.get(
                    "total_validation_buildings_with_custom_roofs"
                ),
                "removed_zero_radiation_buildings_count": summary_dict.get(
                    "removed_zero_radiation_buildings_count"
                ),
                "removed_zero_radiation_buildings": summary_dict.get("removed_zero_radiation_buildings"),
                "matched_buildings": summary_dict.get("matched_buildings"),
                "validation_missing_in_ucea": summary_dict.get("validation_missing_in_ucea"),
                "ucea_only_buildings": summary_dict.get("ucea_only_buildings"),
                "validation_csv_path": summary_dict.get("validation_csv_path"),
                "ucea_summary_csv_path": summary_dict.get("ucea_summary_csv_path"),
                "total_workflow0_roofs_top_m2_matched": summary_dict.get("total_workflow0_roofs_top_m2_matched"),
                "total_workflow0_radiation_kwh_year_matched": summary_dict.get(
                    "total_workflow0_radiation_kwh_year_matched"
                ),
            }
        )
        all_comparison_rows.extend(area.comparison_rows)

    def _sum_numeric(rows: list[dict[str, Any]], key: str) -> float:
        total = 0.0
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                total += float(value)
        return total

    grand_total_rows = [
        {"metric": "areas_total", "value": len(area_comparisons)},
        {"metric": "validation_buildings_total", "value": _sum_numeric(overall_summary_rows, "validation_buildings_total")},
        {
            "metric": "total_validation_buildings_with_custom_roofs",
            "value": _sum_numeric(overall_summary_rows, "total_validation_buildings_with_custom_roofs"),
        },
        {
            "metric": "removed_zero_radiation_buildings_count",
            "value": _sum_numeric(overall_summary_rows, "removed_zero_radiation_buildings_count"),
        },
        {"metric": "matched_buildings", "value": _sum_numeric(overall_summary_rows, "matched_buildings")},
        {
            "metric": "validation_missing_in_ucea",
            "value": _sum_numeric(overall_summary_rows, "validation_missing_in_ucea"),
        },
        {"metric": "ucea_only_buildings", "value": _sum_numeric(overall_summary_rows, "ucea_only_buildings")},
        {
            "metric": "total_validation_area_m2",
            "value": _sum_numeric(all_comparison_rows, "validation_area_m2"),
        },
        {
            "metric": "total_ucea_roofs_top_m2_matched",
            "value": _sum_numeric(all_comparison_rows, "ucea_roofs_top_m2"),
        },
        {
            "metric": "total_workflow0_roofs_top_m2_matched",
            "value": _sum_numeric(all_comparison_rows, "workflow0_roofs_top_m2"),
        },
        {
            "metric": "total_validation_radiation_kwh_year",
            "value": _sum_numeric(all_comparison_rows, "validation_radiation_kwh_year"),
        },
        {
            "metric": "total_ucea_radiation_kwh_year_matched",
            "value": _sum_numeric(all_comparison_rows, "ucea_radiation_kwh_year"),
        },
        {
            "metric": "total_workflow0_radiation_kwh_year_matched",
            "value": _sum_numeric(all_comparison_rows, "workflow0_radiation_kwh_year"),
        },
    ]

    sheets = [
        (
            "area_summary",
            _to_worksheet_table(
                overall_summary_rows,
                [
                    "area_id",
                    "validation_buildings_total",
                    "total_validation_buildings_with_custom_roofs",
                    "removed_zero_radiation_buildings_count",
                    "removed_zero_radiation_buildings",
                    "matched_buildings",
                    "validation_missing_in_ucea",
                    "ucea_only_buildings",
                    "total_workflow0_roofs_top_m2_matched",
                    "total_workflow0_radiation_kwh_year_matched",
                    "validation_csv_path",
                    "ucea_summary_csv_path",
                ],
            ),
        ),
        (
            "totals",
            _to_worksheet_table(grand_total_rows, ["metric", "value"]),
        ),
        (
            "all_comparisons",
            _to_worksheet_table(
                all_comparison_rows,
                [
                    "area_id",
                    "building_name",
                    "status",
                    "validation_area_m2",
                    "ucea_roofs_top_m2",
                    "workflow0_roofs_top_m2",
                    "validation_radiation_kwh_year",
                    "ucea_radiation_kwh_year",
                    "workflow0_radiation_kwh_year",
                    "error_cea",
                    "error_cea_oven",
                    "workflow0_source_column",
                    "workflow0_timesteps",
                    "validation_point_count",
                    "residential_source",
                    "use_type1",
                    "use_type2",
                    "use_type3",
                    "resi_type",
                    "ucea_source_column",
                    "ucea_timesteps",
                    "ucea_oven_confidence_face_count",
                    "ucea_oven_confidence_mean",
                    "ucea_oven_confidence_area_weighted_mean",
                    "ucea_oven_confidence_min",
                    "ucea_oven_confidence_max",
                ],
            ),
        ),
    ]
    _write_xlsx_minimal(output_path, sheets)


def build_comparison_excels(
    validation_root: Path,
    ucea_root: Path,
    output_dir: Path,
    areas_config: Path | None,
) -> None:
    validation_root = _to_absolute(validation_root)
    ucea_root = _to_absolute(ucea_root)
    output_dir = _to_absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_map = _load_area_to_scenario_from_config(_to_absolute(areas_config)) if areas_config else {}
    validation_csvs = _discover_validation_csvs(validation_root)
    if not validation_csvs:
        raise FileNotFoundError(f"No *_residential_radiation.csv files found under: {validation_root}")
    validation_csvs = _deduplicate_validation_csvs_by_area(validation_csvs)

    area_comparisons: list[AreaComparison] = []
    area_error_map_paths: list[Path] = []
    for validation_csv_path in validation_csvs:
        area_id = validation_csv_path.stem.replace("_residential_radiation", "")
        validation_area_dirname = validation_csv_path.parent.name

        scenario_dir = _match_scenario_dir(
            area_id=area_id,
            validation_area_dirname=validation_area_dirname,
            area_to_scenario_from_config=config_map,
            ucea_root=ucea_root,
        )
        ucea_summary_csv_path = scenario_dir / UCEA_ROOFTOP_SUMMARY_RELATIVE if scenario_dir else None
        if ucea_summary_csv_path and not ucea_summary_csv_path.exists():
            ucea_summary_csv_path = None
        workflow0_buildings: dict[str, dict[str, Any]] = {}
        workflow1_buildings: dict[str, dict[str, Any]] = {}
        if scenario_dir:
            workflow0_buildings = _read_workflow0_summary_buildings(scenario_dir / WORKFLOW_0_SOLAR_RELATIVE)
            workflow1_buildings = _read_workflow1_summary_buildings(scenario_dir / WORKFLOW_1_SOLAR_RELATIVE)

        area_comparison = _build_area_comparison(
            area_id=area_id,
            validation_csv_path=validation_csv_path,
            ucea_summary_csv_path=ucea_summary_csv_path,
            workflow0_buildings=workflow0_buildings,
            workflow1_buildings=workflow1_buildings,
        )
        area_comparisons.append(area_comparison)

        area_output_name = f"{_sanitize_filename(area_id)}_radiation_comparison.xlsx"
        area_output_path = output_dir / area_output_name
        _write_area_workbook(area_output_path, area_comparison)
        print(f"Wrote: {area_output_path}")
        residential_geojson_path = _find_validation_residential_geojson(validation_csv_path, area_id)
        error_map_geojson_path = output_dir / f"{_sanitize_filename(area_id)}_radiation_error_map.geojson"
        _write_area_error_map_geojson(error_map_geojson_path, residential_geojson_path, area_comparison)
        area_error_map_paths.append(error_map_geojson_path)
        print(f"Wrote: {error_map_geojson_path}")

    combined_error_map_path = output_dir / "all_areas_radiation_error_map.geojson"
    _write_combined_error_map_geojson(combined_error_map_path, area_error_map_paths)
    print(f"Wrote: {combined_error_map_path}")

    overall_output_path = output_dir / "all_areas_radiation_comparison.xlsx"
    _write_overall_workbook(overall_output_path, area_comparisons)
    print(f"Wrote: {overall_output_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=Path("tools/validation_outputs"),
        help="Root directory containing per-area validation output folders.",
    )
    parser.add_argument(
        "--ucea-root",
        type=Path,
        default=Path(r"C:/Users/Andre/cea-scenarios/Validation"),
        help="Root directory containing UCEA scenario folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tools/comparison_outputs"),
        help="Directory where .xlsx comparison files will be written.",
    )
    parser.add_argument(
        "--areas-config",
        type=Path,
        default=Path("tools/validation_areas_config.json"),
        help="Optional areas config to improve area -> scenario mapping.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    build_comparison_excels(
        validation_root=args.validation_root,
        ucea_root=args.ucea_root,
        output_dir=args.output_dir,
        areas_config=args.areas_config,
    )


if __name__ == "__main__":
    main()
