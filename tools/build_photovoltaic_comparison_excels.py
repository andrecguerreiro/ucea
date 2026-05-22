"""Build per-area photovoltaic comparison workbooks between CEA and UCEA.

Comparison scope is restricted to buildings present in validation residential
datasets (`*_residential_radiation.csv`), which are already filtered by area
and residential type.

Sources:
- CEA  (workflow0): outputs/data/roof-workflow-comparison/
  workflow0_normal_flat_roofs/potentials/solar/PV_<pv_code>_total_buildings.csv
- UCEA (workflow1): outputs/data/roof-workflow-comparison/
  workflow1_geometry_generator/potentials/solar/PV_<pv_code>_total_buildings.csv
- UCEA orientation: workflow1 `*_PV_sensors.csv`
- OVEN metadata: inputs/building-geometry/roof_surfaces.geojson

Outputs (default directory `tools/comparison_outputs/Photovoltaic`):
- One workbook per area: <area_id>_photovoltaic_comparison.xlsx
- One aggregate workbook: all_areas_photovoltaic_comparison.xlsx
- One combined error map: all_areas_photovoltaic_error_map.geojson
- Seasonal PV normalized profiles:
  all_areas_pv_hourly_normalized_shared_denominator_<winter|summer|midseason>.csv
  and all_areas_pv_hourly_normalized_shared_denominator_daily_error_summary.csv
- Seasonal PV normalized profiles without area division:
  all_areas_pv_hourly_normalized_shared_denominator_no_area_<winter|summer|midseason>.csv
  and all_areas_pv_hourly_normalized_shared_denominator_no_area_daily_error_summary.csv
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
from datetime import datetime
from pathlib import Path
from typing import Any


WORKFLOW_COMPARISON_RELATIVE = Path("outputs/data/roof-workflow-comparison")
CEA_WORKFLOW_NAME = "workflow0_normal_flat_roofs"
UCEA_WORKFLOW_NAME = "workflow1_geometry_generator"
ROOF_SURFACES_RELATIVE = Path("inputs/building-geometry/roof_surfaces.geojson")

GLOBAL_FACE_TO_CARDINAL = {
    "T": "NORTH",
    "R": "EAST",
    "B": "SOUTH",
    "L": "WEST",
    "H": "HORIZONTAL",
}
CARDINALS = ("NORTH", "EAST", "SOUTH", "WEST")
SEASON_MONTHS = {
    "winter": {12, 1, 2},
    "summer": {6, 7, 8},
    "midseason": {3, 4, 5, 9, 10, 11},
}


@dataclass
class AreaComparison:
    area_id: str
    validation_csv_path: Path
    scenario_dir: Path | None
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
    invalid = "[]:*?/\\"
    sanitized = "".join("_" if ch in invalid else ch for ch in name).strip()
    sanitized = sanitized or "Sheet"
    return sanitized[:31]


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-\.]+", "_", name)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def _read_geojson(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_geojson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_rows_csv(output_path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header) for header in headers})


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator == 0.0:
        return None
    return numerator / denominator


def _ratio_to_percentage(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value * 100.0


def _safe_subtract(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        return None
    return lhs - rhs


def _normalize_pv_code(raw_code: str) -> str:
    code = (raw_code or "").strip()
    if not code:
        return "PV1"
    if code.upper().startswith("PV_"):
        code = code[3:]
    return code


def _build_pv_total_relative(workflow_name: str, pv_code: str) -> Path:
    return (
        WORKFLOW_COMPARISON_RELATIVE
        / workflow_name
        / "potentials"
        / "solar"
        / f"PV_{pv_code}_total_buildings.csv"
    )


def _build_sensor_dir_relative(workflow_name: str) -> Path:
    return WORKFLOW_COMPARISON_RELATIVE / workflow_name / "potentials" / "solar" / "sensors"


def _discover_validation_csvs(validation_root: Path) -> list[Path]:
    return sorted(validation_root.rglob("*_residential_radiation.csv"))


def _deduplicate_validation_csvs_by_area(validation_csvs: list[Path]) -> list[Path]:
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


def _scenario_has_required_files(scenario_dir: Path, pv_code: str) -> bool:
    cea_total = scenario_dir / _build_pv_total_relative(CEA_WORKFLOW_NAME, pv_code)
    ucea_total = scenario_dir / _build_pv_total_relative(UCEA_WORKFLOW_NAME, pv_code)
    return cea_total.exists() and ucea_total.exists()


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
    pv_code: str,
) -> Path | None:
    config_hit = area_to_scenario_from_config.get(area_id)
    if config_hit and _scenario_has_required_files(config_hit, pv_code):
        return config_hit

    direct = ucea_root / validation_area_dirname
    if _scenario_has_required_files(direct, pv_code):
        return direct

    norm_targets = {_normalize_key(area_id), _normalize_key(validation_area_dirname)}
    for child in sorted([p for p in ucea_root.iterdir() if p.is_dir()]):
        child_norm = _normalize_key(child.name)
        if child_norm in norm_targets and _scenario_has_required_files(child, pv_code):
            return child

    target_tokens = set(_normalize_key(area_id).split()) | set(
        _normalize_key(validation_area_dirname).split()
    )
    for child in sorted([p for p in ucea_root.iterdir() if p.is_dir()]):
        child_tokens = set(_normalize_key(child.name).split())
        if child_tokens and child_tokens.issubset(target_tokens):
            if _scenario_has_required_files(child, pv_code):
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

    for item in result.values():
        if item["validation_point_count"] == 0.0:
            item["validation_point_count"] = None
    return result


def _build_pv_timeseries_dir_relative(workflow_name: str) -> Path:
    return WORKFLOW_COMPARISON_RELATIVE / workflow_name / "potentials" / "solar" / "PV"


def _parse_month_hour_day_from_timestamp(timestamp_text: str) -> tuple[int, int, str] | None:
    text = (timestamp_text or "").strip()
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    if len(text) >= 19:
        candidates.append(text[:19])

    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt.month, dt.hour, dt.date().isoformat()
        except Exception:
            continue

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text[:19], fmt)
            return dt.month, dt.hour, dt.date().isoformat()
        except Exception:
            continue
    return None


def _read_building_hourly_profiles_by_season_from_pv_csv(
    path: Path,
) -> tuple[dict[str, list[float]], float, float | None] | None:
    sums: dict[str, list[float]] = {season: [0.0] * 24 for season in SEASON_MONTHS}
    season_days: dict[str, set[str]] = {season: set() for season in SEASON_MONTHS}
    annual_total_kwh = 0.0
    pv_area_m2: float | None = None

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return None

        date_column = "date" if "date" in fieldnames else ("Date" if "Date" in fieldnames else "")
        if not date_column:
            return None

        energy_column = ""
        if "PV_roofs_top_E_kWh" in fieldnames:
            energy_column = "PV_roofs_top_E_kWh"
        elif "E_PV_gen_kWh" in fieldnames:
            energy_column = "E_PV_gen_kWh"
        else:
            return None

        for row in reader:
            if pv_area_m2 is None:
                pv_area_m2 = _to_float_or_none(row.get("PV_roofs_top_m2"))
                if pv_area_m2 is None:
                    pv_area_m2 = _to_float_or_none(row.get("area_PV_m2"))

            energy_kwh = _to_float_or_none(row.get(energy_column))
            if energy_kwh is None:
                continue

            parsed = _parse_month_hour_day_from_timestamp(str(row.get(date_column) or ""))
            if parsed is None:
                continue
            month, hour, day_id = parsed
            annual_total_kwh += energy_kwh

            for season, months in SEASON_MONTHS.items():
                if month in months:
                    sums[season][hour] += energy_kwh
                    season_days[season].add(day_id)
                    break

    seasonal_profiles: dict[str, list[float]] = {}
    for season in SEASON_MONTHS:
        day_count = len(season_days[season])
        if day_count <= 0:
            seasonal_profiles[season] = [0.0] * 24
            continue
        seasonal_profiles[season] = [value / float(day_count) for value in sums[season]]
    return seasonal_profiles, annual_total_kwh, pv_area_m2


def _read_workflow_pv_hourly_profiles_by_season(
    workflow_pv_dir: Path,
    pv_code: str,
) -> dict[str, tuple[dict[str, list[float]], float, float | None]]:
    if not workflow_pv_dir.exists():
        return {}

    result: dict[str, tuple[dict[str, list[float]], float, float | None]] = {}
    suffix = f"_{pv_code}"
    for path in sorted(workflow_pv_dir.glob(f"*_{pv_code}.csv")):
        stem = path.stem
        building = stem[:-len(suffix)] if stem.endswith(suffix) else stem
        building = building.strip()
        if not building:
            continue
        payload = _read_building_hourly_profiles_by_season_from_pv_csv(path)
        if payload is None:
            continue
        result[building] = payload
    return result


def _normalize_shared_denominator_with_overall_basis(
    profile_cea: list[float],
    profile_ucea: list[float],
    annual_sum_cea: float,
    annual_sum_ucea: float,
    area_cea_m2: float | None,
    area_ucea_m2: float | None,
) -> tuple[list[float], list[float]] | None:
    if area_cea_m2 is None or area_ucea_m2 is None:
        return None
    if area_cea_m2 <= 0.0 or area_ucea_m2 <= 0.0:
        return None

    annual_density_cea = annual_sum_cea / area_cea_m2
    annual_density_ucea = annual_sum_ucea / area_ucea_m2
    denominator = 0.5 * (annual_density_cea + annual_density_ucea)
    if denominator <= 0.0:
        return None
    return (
        [(value / area_cea_m2) / denominator for value in profile_cea],
        [(value / area_ucea_m2) / denominator for value in profile_ucea],
    )


def _normalize_shared_denominator_without_area_basis(
    profile_cea: list[float],
    profile_ucea: list[float],
    annual_sum_cea: float,
    annual_sum_ucea: float,
) -> tuple[list[float], list[float]] | None:
    denominator = 0.5 * (annual_sum_cea + annual_sum_ucea)
    if denominator <= 0.0:
        return None
    return (
        [value / denominator for value in profile_cea],
        [value / denominator for value in profile_ucea],
    )


def _daily_error_between_profiles(profile_cea: list[float], profile_ucea: list[float]) -> float:
    return sum(abs(profile_cea[h] - profile_ucea[h]) for h in range(24))


def _write_hourly_normalized_shared_denominator_csv(
    output_csv_path: Path,
    profile_pairs: list[tuple[list[float], list[float]]],
) -> None:
    if not profile_pairs:
        raise ValueError(f"No common PV building profiles available for: {output_csv_path.name}")

    total = float(len(profile_pairs))
    mean_cea = []
    mean_ucea = []
    for hour in range(24):
        mean_cea.append(sum(pair[0][hour] for pair in profile_pairs) / total)
        mean_ucea.append(sum(pair[1][hour] for pair in profile_pairs) / total)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["hour", "cea", "ucea"])
        for hour in range(24):
            writer.writerow([hour, mean_cea[hour], mean_ucea[hour]])


def _write_daily_error_summary_csv(
    output_csv_path: Path,
    daily_errors_by_season: dict[str, list[float]],
) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["season", "avg_total_daily_error_cea_vs_ucea"])
        for season in ("winter", "summer", "midseason"):
            values = daily_errors_by_season.get(season, [])
            if not values:
                raise ValueError(f"No daily error values available for season '{season}'.")
            writer.writerow([season, sum(values) / float(len(values))])


def _write_pv_hourly_normalized_shared_denominator_csvs_by_season(
    area_comparisons: list[AreaComparison],
    output_csv_prefix: Path,
    pv_code: str,
    divide_by_area: bool,
) -> None:
    all_profile_pairs_by_season: dict[str, list[tuple[list[float], list[float]]]] = {
        season: [] for season in SEASON_MONTHS
    }
    daily_errors_by_season: dict[str, list[float]] = {season: [] for season in SEASON_MONTHS}
    mode_label = "area-normalized" if divide_by_area else "no-area-normalized"

    for area in area_comparisons:
        scenario_dir = area.scenario_dir
        if scenario_dir is None:
            print(f"[warn] Skipping PV seasonal profile for area '{area.area_id}': scenario not resolved.")
            continue

        validation_buildings = {
            str(row.get("building_name") or "").strip() for row in area.comparison_rows if row.get("building_name")
        }
        if not validation_buildings:
            print(f"[warn] Skipping PV seasonal profile for area '{area.area_id}': no validation buildings.")
            continue

        workflow0_profiles = _read_workflow_pv_hourly_profiles_by_season(
            scenario_dir / _build_pv_timeseries_dir_relative(CEA_WORKFLOW_NAME),
            pv_code,
        )
        workflow1_profiles = _read_workflow_pv_hourly_profiles_by_season(
            scenario_dir / _build_pv_timeseries_dir_relative(UCEA_WORKFLOW_NAME),
            pv_code,
        )
        if not workflow0_profiles or not workflow1_profiles:
            print(
                f"[warn] Skipping PV seasonal profile for area '{area.area_id}': "
                "missing workflow0/workflow1 PV timeseries."
            )
            continue

        common_buildings = sorted(set(workflow0_profiles).intersection(workflow1_profiles).intersection(validation_buildings))
        if not common_buildings:
            print(
                f"[warn] Skipping PV seasonal profile for area '{area.area_id}': "
                "no common validation buildings between workflows."
            )
            continue

        area_used_buildings = 0
        for building in common_buildings:
            wf0_profiles_by_season, wf0_annual_sum, wf0_area_m2 = workflow0_profiles[building]
            wf1_profiles_by_season, wf1_annual_sum, wf1_area_m2 = workflow1_profiles[building]
            contributed = False
            for season in SEASON_MONTHS:
                if divide_by_area:
                    normalized = _normalize_shared_denominator_with_overall_basis(
                        wf0_profiles_by_season[season],
                        wf1_profiles_by_season[season],
                        wf0_annual_sum,
                        wf1_annual_sum,
                        wf0_area_m2,
                        wf1_area_m2,
                    )
                else:
                    normalized = _normalize_shared_denominator_without_area_basis(
                        wf0_profiles_by_season[season],
                        wf1_profiles_by_season[season],
                        wf0_annual_sum,
                        wf1_annual_sum,
                    )
                if normalized is None:
                    continue
                all_profile_pairs_by_season[season].append(normalized)
                daily_errors_by_season[season].append(_daily_error_between_profiles(normalized[0], normalized[1]))
                contributed = True
            if contributed:
                area_used_buildings += 1

        print(
            f"[info] PV seasonal area '{area.area_id}' ({mode_label}): "
            f"common_validation_buildings={len(common_buildings)}, used_buildings={area_used_buildings}"
        )

    for season in ("winter", "summer", "midseason"):
        season_output_path = output_csv_prefix.parent / f"{output_csv_prefix.name}_{season}.csv"
        _write_hourly_normalized_shared_denominator_csv(
            season_output_path,
            all_profile_pairs_by_season[season],
        )
        print(f"Wrote: {season_output_path}")
        print(
            f"[info] Total PV buildings used across all areas ({season}, {mode_label}): "
            f"{len(all_profile_pairs_by_season[season])}"
        )

    summary_output_path = output_csv_prefix.parent / f"{output_csv_prefix.name}_daily_error_summary.csv"
    _write_daily_error_summary_csv(summary_output_path, daily_errors_by_season)
    print(f"Wrote: {summary_output_path}")


def _read_pv_totals_by_building(path: Path, prefix: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows = _read_csv_rows(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        building = str(row.get("name") or "").strip()
        if not building:
            continue

        area_m2 = _to_float_or_none(row.get("PV_roofs_top_m2"))
        if area_m2 is None:
            area_m2 = _to_float_or_none(row.get("area_PV_m2"))

        generation_kwh = _to_float_or_none(row.get("PV_roofs_top_E_kWh"))
        if generation_kwh is None:
            generation_kwh = _to_float_or_none(row.get("E_PV_gen_kWh"))

        result[building] = {
            f"{prefix}_AREA": area_m2,
            f"{prefix}_GENERATION_KWH_YEAR": generation_kwh,
        }
    return result


def _azimuth_to_cardinal(azimuth_deg: float) -> str | None:
    if not math.isfinite(azimuth_deg):
        return None
    angle = azimuth_deg % 360.0
    if angle >= 315.0 or angle < 45.0:
        return "NORTH"
    if angle >= 45.0 and angle < 135.0:
        return "EAST"
    if angle >= 135.0 and angle < 225.0:
        return "SOUTH"
    if angle >= 225.0 and angle < 315.0:
        return "WEST"
    return None


def _read_ucea_orientation_from_sensors(sensor_dir: Path) -> dict[str, dict[str, Any]]:
    if not sensor_dir.exists():
        return {}

    result: dict[str, dict[str, Any]] = {}
    for path in sorted(sensor_dir.glob("*_PV_sensors.csv")):
        building_from_name = path.stem.replace("_PV_sensors", "").strip()

        area_by_cardinal = {k: 0.0 for k in CARDINALS}
        weight_by_cardinal = {k: 0.0 for k in CARDINALS}
        total_area = 0.0
        total_weight = 0.0
        building_id = building_from_name

        with path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                building_field = str(row.get("BUILDING") or "").strip()
                if building_field:
                    building_id = building_field

                area = _to_float_or_none(row.get("area_installed_module_m2"))
                if area is None:
                    area = _to_float_or_none(row.get("AREA_m2"))
                azimuth = _to_float_or_none(row.get("surface_azimuth_deg"))
                total_rad_wh_m2 = _to_float_or_none(row.get("total_rad_Whm2"))

                if area is None or area <= 0.0 or azimuth is None:
                    continue
                cardinal = _azimuth_to_cardinal(azimuth)
                if cardinal is None:
                    continue

                area_by_cardinal[cardinal] += area
                total_area += area

                if total_rad_wh_m2 is not None and total_rad_wh_m2 > 0.0:
                    weight = area * total_rad_wh_m2
                    weight_by_cardinal[cardinal] += weight
                    total_weight += weight

        if not building_id:
            continue

        payload: dict[str, Any] = {}
        for cardinal in CARDINALS:
            area_pct = _ratio_to_percentage(_safe_divide(area_by_cardinal[cardinal], total_area))
            gen_pct = _ratio_to_percentage(_safe_divide(weight_by_cardinal[cardinal], total_weight))
            payload[f"{cardinal}_PCT_AREA"] = area_pct
            payload[f"{cardinal}_PCT_GEN"] = gen_pct
            payload[f"UCEA_{cardinal}_AREA_M2"] = area_by_cardinal[cardinal]
            payload[f"UCEA_{cardinal}_GEN_WEIGHT"] = weight_by_cardinal[cardinal]

        payload["UCEA_SENSOR_TOTAL_AREA_M2"] = total_area if total_area > 0.0 else None
        payload["UCEA_SENSOR_TOTAL_GEN_WEIGHT"] = total_weight if total_weight > 0.0 else None
        result[building_id] = payload

    return result


def _read_oven_roof_metadata(roof_surfaces_path: Path) -> dict[str, dict[str, Any]]:
    if not roof_surfaces_path.exists():
        return {}

    payload = _read_geojson(roof_surfaces_path)
    features = payload.get("features", [])

    result: dict[str, dict[str, Any]] = {}
    confidence_values_by_building: dict[str, list[tuple[float, float | None]]] = {}
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties", {})
        building = str((props or {}).get("building") or "").strip()
        if not building:
            continue

        roof_face_global = str((props or {}).get("roof_face_global") or "").strip().upper()
        cardinal = GLOBAL_FACE_TO_CARDINAL.get(roof_face_global)

        if building not in result:
            result[building] = {
                "OVEN_ROOF_DETECTED": 0,
                "OVEN_ROOF_COUNT": 0,
                "OVEN_COUNT_NORTH": 0,
                "OVEN_COUNT_EAST": 0,
                "OVEN_COUNT_SOUTH": 0,
                "OVEN_COUNT_WEST": 0,
                "OVEN_COUNT_HORIZONTAL": 0,
                "OVEN_CONFIDENCE_FACE_COUNT": 0,
                "OVEN_CONFIDENCE_MEAN": None,
                "OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN": None,
                "OVEN_CONFIDENCE_MIN": None,
                "OVEN_CONFIDENCE_MAX": None,
            }

        result[building]["OVEN_ROOF_COUNT"] += 1
        if cardinal == "NORTH":
            result[building]["OVEN_COUNT_NORTH"] += 1
        elif cardinal == "EAST":
            result[building]["OVEN_COUNT_EAST"] += 1
        elif cardinal == "SOUTH":
            result[building]["OVEN_COUNT_SOUTH"] += 1
        elif cardinal == "WEST":
            result[building]["OVEN_COUNT_WEST"] += 1
        elif cardinal == "HORIZONTAL":
            result[building]["OVEN_COUNT_HORIZONTAL"] += 1

        confidence = _to_float_or_none((props or {}).get("roof_confidence"))
        if confidence is not None and math.isfinite(confidence):
            area_m2 = _to_float_or_none((props or {}).get("roof_area_m2"))
            confidence_values_by_building.setdefault(building, []).append((float(confidence), area_m2))

    for building in result:
        result[building]["OVEN_ROOF_DETECTED"] = 1 if result[building]["OVEN_ROOF_COUNT"] > 0 else 0
        confidence_values = confidence_values_by_building.get(building, [])
        confidences = [confidence for confidence, _ in confidence_values if math.isfinite(confidence)]
        if not confidences:
            continue

        weighted_values = [
            (confidence, area_m2)
            for confidence, area_m2 in confidence_values
            if area_m2 is not None and math.isfinite(area_m2) and area_m2 > 0.0
        ]
        weighted_mean: float | None = None
        if weighted_values:
            total_area = sum(area_m2 for _, area_m2 in weighted_values)
            if total_area > 0.0:
                weighted_mean = sum(confidence * area_m2 for confidence, area_m2 in weighted_values) / total_area

        result[building]["OVEN_CONFIDENCE_FACE_COUNT"] = int(len(confidences))
        result[building]["OVEN_CONFIDENCE_MEAN"] = float(sum(confidences) / len(confidences))
        result[building]["OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN"] = weighted_mean
        result[building]["OVEN_CONFIDENCE_MIN"] = float(min(confidences))
        result[building]["OVEN_CONFIDENCE_MAX"] = float(max(confidences))

    return result


def _find_validation_residential_geojson(validation_csv_path: Path, area_id: str) -> Path | None:
    candidate = validation_csv_path.parent / f"{area_id}_residential_buildings.geojson"
    if candidate.exists():
        return candidate
    return None


def _comparison_headers() -> list[str]:
    return [
        "area_id",
        "building_name",
        "status",
        "CEA_AREA",
        "UCEA_AREA",
        "CEA_GENERATION_KWH_YEAR",
        "UCEA_GENERATION_KWH_YEAR",
        "UCEA_MINUS_CEA_KWH_YEAR",
        "UCEA_VS_CEA_RATIO",
        "UCEA_VS_CEA_PCT_CHANGE",
        "NORTH_PCT_AREA",
        "EAST_PCT_AREA",
        "SOUTH_PCT_AREA",
        "WEST_PCT_AREA",
        "NORTH_PCT_GEN",
        "EAST_PCT_GEN",
        "SOUTH_PCT_GEN",
        "WEST_PCT_GEN",
        "OVEN_ROOF_DETECTED",
        "OVEN_ROOF_COUNT",
        "OVEN_COUNT_NORTH",
        "OVEN_COUNT_EAST",
        "OVEN_COUNT_SOUTH",
        "OVEN_COUNT_WEST",
        "OVEN_COUNT_HORIZONTAL",
        "OVEN_CONFIDENCE_FACE_COUNT",
        "OVEN_CONFIDENCE_MEAN",
        "OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN",
        "OVEN_CONFIDENCE_MIN",
        "OVEN_CONFIDENCE_MAX",
        "validation_area_m2",
        "validation_radiation_kwh_year",
        "validation_point_count",
        "residential_source",
        "use_type1",
        "use_type2",
        "use_type3",
        "resi_type",
    ]


def _build_area_comparison(
    area_id: str,
    validation_csv_path: Path,
    scenario_dir: Path | None,
    pv_code: str,
) -> AreaComparison:
    validation_rows = _read_csv_rows(validation_csv_path)
    validation_buildings = _aggregate_validation_buildings(validation_rows)

    cea_buildings: dict[str, dict[str, Any]] = {}
    ucea_buildings: dict[str, dict[str, Any]] = {}
    ucea_orientation: dict[str, dict[str, Any]] = {}
    oven_metadata: dict[str, dict[str, Any]] = {}

    if scenario_dir is not None:
        cea_total_path = scenario_dir / _build_pv_total_relative(CEA_WORKFLOW_NAME, pv_code)
        ucea_total_path = scenario_dir / _build_pv_total_relative(UCEA_WORKFLOW_NAME, pv_code)
        ucea_sensor_dir = scenario_dir / _build_sensor_dir_relative(UCEA_WORKFLOW_NAME)
        roof_surfaces_path = scenario_dir / ROOF_SURFACES_RELATIVE

        cea_buildings = _read_pv_totals_by_building(cea_total_path, "CEA")
        ucea_buildings = _read_pv_totals_by_building(ucea_total_path, "UCEA")
        ucea_orientation = _read_ucea_orientation_from_sensors(ucea_sensor_dir)
        oven_metadata = _read_oven_roof_metadata(roof_surfaces_path)

    comparison_rows: list[dict[str, Any]] = []
    validation_missing_in_ucea_rows: list[dict[str, Any]] = []

    for building in sorted(validation_buildings):
        v = validation_buildings[building]
        c = cea_buildings.get(building, {})
        u = ucea_buildings.get(building, {})
        orient = ucea_orientation.get(building, {})
        oven = oven_metadata.get(building, {})

        cea_area = _to_float_or_none(c.get("CEA_AREA"))
        ucea_area = _to_float_or_none(u.get("UCEA_AREA"))
        cea_generation = _to_float_or_none(c.get("CEA_GENERATION_KWH_YEAR"))
        ucea_generation = _to_float_or_none(u.get("UCEA_GENERATION_KWH_YEAR"))
        ucea_minus_cea = _safe_subtract(ucea_generation, cea_generation)
        ucea_vs_cea_ratio = _safe_divide(ucea_minus_cea, cea_generation)
        ucea_vs_cea_pct_change = _ratio_to_percentage(ucea_vs_cea_ratio)

        if u:
            status = "matched" if c else "missing_in_cea"
        else:
            status = "missing_in_ucea"

        row = {
            "area_id": area_id,
            "building_name": building,
            "status": status,
            "CEA_AREA": cea_area,
            "UCEA_AREA": ucea_area,
            "CEA_GENERATION_KWH_YEAR": cea_generation,
            "UCEA_GENERATION_KWH_YEAR": ucea_generation,
            "UCEA_MINUS_CEA_KWH_YEAR": ucea_minus_cea,
            "UCEA_VS_CEA_RATIO": ucea_vs_cea_ratio,
            "UCEA_VS_CEA_PCT_CHANGE": ucea_vs_cea_pct_change,
            "NORTH_PCT_AREA": orient.get("NORTH_PCT_AREA"),
            "EAST_PCT_AREA": orient.get("EAST_PCT_AREA"),
            "SOUTH_PCT_AREA": orient.get("SOUTH_PCT_AREA"),
            "WEST_PCT_AREA": orient.get("WEST_PCT_AREA"),
            "NORTH_PCT_GEN": orient.get("NORTH_PCT_GEN"),
            "EAST_PCT_GEN": orient.get("EAST_PCT_GEN"),
            "SOUTH_PCT_GEN": orient.get("SOUTH_PCT_GEN"),
            "WEST_PCT_GEN": orient.get("WEST_PCT_GEN"),
            "OVEN_ROOF_DETECTED": oven.get("OVEN_ROOF_DETECTED", 0),
            "OVEN_ROOF_COUNT": oven.get("OVEN_ROOF_COUNT", 0),
            "OVEN_COUNT_NORTH": oven.get("OVEN_COUNT_NORTH", 0),
            "OVEN_COUNT_EAST": oven.get("OVEN_COUNT_EAST", 0),
            "OVEN_COUNT_SOUTH": oven.get("OVEN_COUNT_SOUTH", 0),
            "OVEN_COUNT_WEST": oven.get("OVEN_COUNT_WEST", 0),
            "OVEN_COUNT_HORIZONTAL": oven.get("OVEN_COUNT_HORIZONTAL", 0),
            "OVEN_CONFIDENCE_FACE_COUNT": oven.get("OVEN_CONFIDENCE_FACE_COUNT", 0),
            "OVEN_CONFIDENCE_MEAN": oven.get("OVEN_CONFIDENCE_MEAN"),
            "OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN": oven.get("OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN"),
            "OVEN_CONFIDENCE_MIN": oven.get("OVEN_CONFIDENCE_MIN"),
            "OVEN_CONFIDENCE_MAX": oven.get("OVEN_CONFIDENCE_MAX"),
            "validation_area_m2": v.get("validation_area_m2"),
            "validation_radiation_kwh_year": v.get("validation_radiation_kwh_year"),
            "validation_point_count": v.get("validation_point_count"),
            "residential_source": v.get("residential_source"),
            "use_type1": v.get("use_type1"),
            "use_type2": v.get("use_type2"),
            "use_type3": v.get("use_type3"),
            "resi_type": v.get("resi_type"),
        }
        comparison_rows.append(row)
        if status == "missing_in_ucea":
            validation_missing_in_ucea_rows.append(row)

    ucea_only_rows: list[dict[str, Any]] = []
    for building in sorted(set(ucea_buildings) - set(validation_buildings)):
        c = cea_buildings.get(building, {})
        u = ucea_buildings.get(building, {})
        orient = ucea_orientation.get(building, {})
        oven = oven_metadata.get(building, {})

        cea_generation = _to_float_or_none(c.get("CEA_GENERATION_KWH_YEAR"))
        ucea_generation = _to_float_or_none(u.get("UCEA_GENERATION_KWH_YEAR"))
        ucea_minus_cea = _safe_subtract(ucea_generation, cea_generation)
        ucea_vs_cea_ratio = _safe_divide(ucea_minus_cea, cea_generation)
        ucea_vs_cea_pct_change = _ratio_to_percentage(ucea_vs_cea_ratio)

        ucea_only_rows.append(
            {
                "area_id": area_id,
                "building_name": building,
                "status": "ucea_only",
                "CEA_AREA": _to_float_or_none(c.get("CEA_AREA")),
                "UCEA_AREA": _to_float_or_none(u.get("UCEA_AREA")),
                "CEA_GENERATION_KWH_YEAR": cea_generation,
                "UCEA_GENERATION_KWH_YEAR": ucea_generation,
                "UCEA_MINUS_CEA_KWH_YEAR": ucea_minus_cea,
                "UCEA_VS_CEA_RATIO": ucea_vs_cea_ratio,
                "UCEA_VS_CEA_PCT_CHANGE": ucea_vs_cea_pct_change,
                "NORTH_PCT_AREA": orient.get("NORTH_PCT_AREA"),
                "EAST_PCT_AREA": orient.get("EAST_PCT_AREA"),
                "SOUTH_PCT_AREA": orient.get("SOUTH_PCT_AREA"),
                "WEST_PCT_AREA": orient.get("WEST_PCT_AREA"),
                "NORTH_PCT_GEN": orient.get("NORTH_PCT_GEN"),
                "EAST_PCT_GEN": orient.get("EAST_PCT_GEN"),
                "SOUTH_PCT_GEN": orient.get("SOUTH_PCT_GEN"),
                "WEST_PCT_GEN": orient.get("WEST_PCT_GEN"),
                "OVEN_ROOF_DETECTED": oven.get("OVEN_ROOF_DETECTED", 0),
                "OVEN_ROOF_COUNT": oven.get("OVEN_ROOF_COUNT", 0),
                "OVEN_COUNT_NORTH": oven.get("OVEN_COUNT_NORTH", 0),
                "OVEN_COUNT_EAST": oven.get("OVEN_COUNT_EAST", 0),
                "OVEN_COUNT_SOUTH": oven.get("OVEN_COUNT_SOUTH", 0),
                "OVEN_COUNT_WEST": oven.get("OVEN_COUNT_WEST", 0),
                "OVEN_COUNT_HORIZONTAL": oven.get("OVEN_COUNT_HORIZONTAL", 0),
                "OVEN_CONFIDENCE_FACE_COUNT": oven.get("OVEN_CONFIDENCE_FACE_COUNT", 0),
                "OVEN_CONFIDENCE_MEAN": oven.get("OVEN_CONFIDENCE_MEAN"),
                "OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN": oven.get("OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN"),
                "OVEN_CONFIDENCE_MIN": oven.get("OVEN_CONFIDENCE_MIN"),
                "OVEN_CONFIDENCE_MAX": oven.get("OVEN_CONFIDENCE_MAX"),
                "validation_area_m2": None,
                "validation_radiation_kwh_year": None,
                "validation_point_count": None,
                "residential_source": "",
                "use_type1": "",
                "use_type2": "",
                "use_type3": "",
                "resi_type": "",
            }
        )

    def _sum_numeric(rows: list[dict[str, Any]], key: str) -> float:
        total = 0.0
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                total += float(value)
        return total

    matched_rows = [r for r in comparison_rows if r.get("status") == "matched"]
    total_cea_generation = _sum_numeric(matched_rows, "CEA_GENERATION_KWH_YEAR")
    total_ucea_generation = _sum_numeric(matched_rows, "UCEA_GENERATION_KWH_YEAR")
    total_delta_generation = total_ucea_generation - total_cea_generation
    total_pct_change = _ratio_to_percentage(_safe_divide(total_delta_generation, total_cea_generation))
    roof_detected_buildings = sum(1 for r in comparison_rows if int(r.get("OVEN_ROOF_DETECTED") or 0) == 1)

    summary_rows = [
        {"metric": "area_id", "value": area_id},
        {"metric": "validation_csv_path", "value": str(validation_csv_path)},
        {"metric": "scenario_dir", "value": str(scenario_dir) if scenario_dir else ""},
        {"metric": "pv_code", "value": pv_code},
        {"metric": "validation_buildings_total", "value": len(comparison_rows)},
        {"metric": "matched_buildings", "value": len(matched_rows)},
        {"metric": "validation_missing_in_ucea", "value": len(validation_missing_in_ucea_rows)},
        {"metric": "ucea_only_buildings", "value": len(ucea_only_rows)},
        {"metric": "roof_detected_buildings", "value": roof_detected_buildings},
        {"metric": "total_cea_generation_kwh_year_matched", "value": total_cea_generation},
        {"metric": "total_ucea_generation_kwh_year_matched", "value": total_ucea_generation},
        {"metric": "total_ucea_minus_cea_kwh_year_matched", "value": total_delta_generation},
        {"metric": "total_ucea_vs_cea_pct_change_matched", "value": total_pct_change},
    ]

    return AreaComparison(
        area_id=area_id,
        validation_csv_path=validation_csv_path,
        scenario_dir=scenario_dir,
        comparison_rows=comparison_rows,
        validation_missing_in_ucea_rows=validation_missing_in_ucea_rows,
        ucea_only_rows=ucea_only_rows,
        summary_rows=summary_rows,
    )


def _augment_feature_for_error_map(feature: dict[str, Any], comparison_row: dict[str, Any]) -> dict[str, Any]:
    props = dict(feature.get("properties", {}))
    cea_gen = _to_float_or_none(comparison_row.get("CEA_GENERATION_KWH_YEAR"))
    ucea_gen = _to_float_or_none(comparison_row.get("UCEA_GENERATION_KWH_YEAR"))
    delta = _safe_subtract(ucea_gen, cea_gen)
    ratio = _safe_divide(delta, cea_gen)
    pct = _ratio_to_percentage(ratio)

    props.update(
        {
            "area_id": comparison_row.get("area_id"),
            "building_name": comparison_row.get("building_name"),
            "comparison_status": comparison_row.get("status"),
            "CEA_GENERATION_KWH_YEAR": cea_gen,
            "UCEA_GENERATION_KWH_YEAR": ucea_gen,
            "UCEA_MINUS_CEA_KWH_YEAR": delta,
            "UCEA_VS_CEA_RATIO": ratio,
            "UCEA_VS_CEA_PCT_CHANGE": pct,
            "UCEA_VS_CEA_ABS_PCT_CHANGE": abs(pct) if pct is not None else None,
            "CEA_AREA": _to_float_or_none(comparison_row.get("CEA_AREA")),
            "UCEA_AREA": _to_float_or_none(comparison_row.get("UCEA_AREA")),
            "NORTH_PCT_AREA": comparison_row.get("NORTH_PCT_AREA"),
            "EAST_PCT_AREA": comparison_row.get("EAST_PCT_AREA"),
            "SOUTH_PCT_AREA": comparison_row.get("SOUTH_PCT_AREA"),
            "WEST_PCT_AREA": comparison_row.get("WEST_PCT_AREA"),
            "NORTH_PCT_GEN": comparison_row.get("NORTH_PCT_GEN"),
            "EAST_PCT_GEN": comparison_row.get("EAST_PCT_GEN"),
            "SOUTH_PCT_GEN": comparison_row.get("SOUTH_PCT_GEN"),
            "WEST_PCT_GEN": comparison_row.get("WEST_PCT_GEN"),
            "OVEN_ROOF_DETECTED": comparison_row.get("OVEN_ROOF_DETECTED"),
            "OVEN_ROOF_COUNT": comparison_row.get("OVEN_ROOF_COUNT"),
            "OVEN_COUNT_NORTH": comparison_row.get("OVEN_COUNT_NORTH"),
            "OVEN_COUNT_EAST": comparison_row.get("OVEN_COUNT_EAST"),
            "OVEN_COUNT_SOUTH": comparison_row.get("OVEN_COUNT_SOUTH"),
            "OVEN_COUNT_WEST": comparison_row.get("OVEN_COUNT_WEST"),
            "OVEN_COUNT_HORIZONTAL": comparison_row.get("OVEN_COUNT_HORIZONTAL"),
            "OVEN_CONFIDENCE_FACE_COUNT": comparison_row.get("OVEN_CONFIDENCE_FACE_COUNT"),
            "OVEN_CONFIDENCE_MEAN": comparison_row.get("OVEN_CONFIDENCE_MEAN"),
            "OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN": comparison_row.get("OVEN_CONFIDENCE_AREA_WEIGHTED_MEAN"),
            "OVEN_CONFIDENCE_MIN": comparison_row.get("OVEN_CONFIDENCE_MIN"),
            "OVEN_CONFIDENCE_MAX": comparison_row.get("OVEN_CONFIDENCE_MAX"),
        }
    )
    return {
        "type": feature.get("type", "Feature"),
        "properties": props,
        "geometry": feature.get("geometry"),
    }


def _write_combined_error_map_geojson(output_path: Path, area_comparisons: list[AreaComparison]) -> None:
    out_features: list[dict[str, Any]] = []
    for area in area_comparisons:
        residential_geojson_path = _find_validation_residential_geojson(area.validation_csv_path, area.area_id)
        if residential_geojson_path is None or not residential_geojson_path.exists():
            continue

        payload = _read_geojson(residential_geojson_path)
        if payload.get("type") != "FeatureCollection":
            continue

        comparison_by_building = {
            str(row.get("building_name") or "").strip(): row for row in area.comparison_rows
        }
        for feature in payload.get("features", []):
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
        "name": "all_areas_photovoltaic_error_map",
        "features": out_features,
    }
    _write_geojson(output_path, out_payload)


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
        '<sheetFormatPr defaultRowHeight="15"/>'
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
    headers = _comparison_headers()

    summary_table = _to_worksheet_table(area_comparison.summary_rows, ["metric", "value"])
    comparison_table = _to_worksheet_table(area_comparison.comparison_rows, headers)
    missing_table = _to_worksheet_table(area_comparison.validation_missing_in_ucea_rows, headers)
    ucea_only_table = _to_worksheet_table(area_comparison.ucea_only_rows, headers)

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
                "matched_buildings": summary_dict.get("matched_buildings"),
                "validation_missing_in_ucea": summary_dict.get("validation_missing_in_ucea"),
                "ucea_only_buildings": summary_dict.get("ucea_only_buildings"),
                "roof_detected_buildings": summary_dict.get("roof_detected_buildings"),
                "total_cea_generation_kwh_year_matched": summary_dict.get(
                    "total_cea_generation_kwh_year_matched"
                ),
                "total_ucea_generation_kwh_year_matched": summary_dict.get(
                    "total_ucea_generation_kwh_year_matched"
                ),
                "total_ucea_minus_cea_kwh_year_matched": summary_dict.get(
                    "total_ucea_minus_cea_kwh_year_matched"
                ),
                "total_ucea_vs_cea_pct_change_matched": summary_dict.get(
                    "total_ucea_vs_cea_pct_change_matched"
                ),
                "validation_csv_path": summary_dict.get("validation_csv_path"),
                "scenario_dir": summary_dict.get("scenario_dir"),
                "pv_code": summary_dict.get("pv_code"),
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

    matched_rows = [r for r in all_comparison_rows if r.get("status") == "matched"]
    total_cea_generation = _sum_numeric(matched_rows, "CEA_GENERATION_KWH_YEAR")
    total_ucea_generation = _sum_numeric(matched_rows, "UCEA_GENERATION_KWH_YEAR")
    total_delta_generation = total_ucea_generation - total_cea_generation
    total_pct_change = _ratio_to_percentage(_safe_divide(total_delta_generation, total_cea_generation))

    grand_total_rows = [
        {"metric": "areas_total", "value": len(area_comparisons)},
        {
            "metric": "validation_buildings_total",
            "value": _sum_numeric(overall_summary_rows, "validation_buildings_total"),
        },
        {"metric": "matched_buildings", "value": _sum_numeric(overall_summary_rows, "matched_buildings")},
        {
            "metric": "validation_missing_in_ucea",
            "value": _sum_numeric(overall_summary_rows, "validation_missing_in_ucea"),
        },
        {"metric": "ucea_only_buildings", "value": _sum_numeric(overall_summary_rows, "ucea_only_buildings")},
        {
            "metric": "roof_detected_buildings",
            "value": _sum_numeric(overall_summary_rows, "roof_detected_buildings"),
        },
        {"metric": "total_cea_generation_kwh_year_matched", "value": total_cea_generation},
        {"metric": "total_ucea_generation_kwh_year_matched", "value": total_ucea_generation},
        {"metric": "total_ucea_minus_cea_kwh_year_matched", "value": total_delta_generation},
        {"metric": "total_ucea_vs_cea_pct_change_matched", "value": total_pct_change},
    ]

    sheets = [
        (
            "area_summary",
            _to_worksheet_table(
                overall_summary_rows,
                [
                    "area_id",
                    "validation_buildings_total",
                    "matched_buildings",
                    "validation_missing_in_ucea",
                    "ucea_only_buildings",
                    "roof_detected_buildings",
                    "total_cea_generation_kwh_year_matched",
                    "total_ucea_generation_kwh_year_matched",
                    "total_ucea_minus_cea_kwh_year_matched",
                    "total_ucea_vs_cea_pct_change_matched",
                    "validation_csv_path",
                    "scenario_dir",
                    "pv_code",
                ],
            ),
        ),
        ("totals", _to_worksheet_table(grand_total_rows, ["metric", "value"])),
        ("all_comparisons", _to_worksheet_table(all_comparison_rows, _comparison_headers())),
    ]
    _write_xlsx_minimal(output_path, sheets)


def build_photovoltaic_comparison_excels(
    validation_root: Path,
    ucea_root: Path,
    output_dir: Path,
    areas_config: Path | None,
    pv_code: str,
    hourly_normalized_shared_prefix: Path | None,
    hourly_normalized_shared_no_area_prefix: Path | None,
) -> None:
    validation_root = _to_absolute(validation_root)
    ucea_root = _to_absolute(ucea_root)
    output_dir = _to_absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pv_code = _normalize_pv_code(pv_code)
    if hourly_normalized_shared_prefix is None:
        hourly_normalized_shared_prefix = output_dir / "all_areas_pv_hourly_normalized_shared_denominator"
    else:
        hourly_normalized_shared_prefix = _to_absolute(hourly_normalized_shared_prefix)
    if hourly_normalized_shared_no_area_prefix is None:
        hourly_normalized_shared_no_area_prefix = (
            output_dir / "all_areas_pv_hourly_normalized_shared_denominator_no_area"
        )
    else:
        hourly_normalized_shared_no_area_prefix = _to_absolute(hourly_normalized_shared_no_area_prefix)

    config_map = _load_area_to_scenario_from_config(_to_absolute(areas_config)) if areas_config else {}
    validation_csvs = _discover_validation_csvs(validation_root)
    if not validation_csvs:
        raise FileNotFoundError(f"No *_residential_radiation.csv files found under: {validation_root}")
    validation_csvs = _deduplicate_validation_csvs_by_area(validation_csvs)

    area_comparisons: list[AreaComparison] = []
    all_areas_comparison_rows: list[dict[str, Any]] = []
    comparison_headers = _comparison_headers()
    for validation_csv_path in validation_csvs:
        area_id = validation_csv_path.stem.replace("_residential_radiation", "")
        validation_area_dirname = validation_csv_path.parent.name

        scenario_dir = _match_scenario_dir(
            area_id=area_id,
            validation_area_dirname=validation_area_dirname,
            area_to_scenario_from_config=config_map,
            ucea_root=ucea_root,
            pv_code=pv_code,
        )
        if scenario_dir is None:
            print(f"[warn] No matching scenario with required PV files for area '{area_id}'.")

        area_comparison = _build_area_comparison(
            area_id=area_id,
            validation_csv_path=validation_csv_path,
            scenario_dir=scenario_dir,
            pv_code=pv_code,
        )
        area_comparisons.append(area_comparison)

        area_output_name = f"{_sanitize_filename(area_id)}_photovoltaic_comparison.xlsx"
        area_output_path = output_dir / area_output_name
        _write_area_workbook(area_output_path, area_comparison)
        print(f"Wrote: {area_output_path}")

        area_comparison_csv_path = output_dir / f"{_sanitize_filename(area_id)}_photovoltaic_comparison.csv"
        _write_rows_csv(area_comparison_csv_path, area_comparison.comparison_rows, comparison_headers)
        print(f"Wrote: {area_comparison_csv_path}")

        all_areas_comparison_rows.extend(area_comparison.comparison_rows)

    combined_error_map_path = output_dir / "all_areas_photovoltaic_error_map.geojson"
    _write_combined_error_map_geojson(combined_error_map_path, area_comparisons)
    print(f"Wrote: {combined_error_map_path}")

    overall_output_path = output_dir / "all_areas_photovoltaic_comparison.xlsx"
    _write_overall_workbook(overall_output_path, area_comparisons)
    print(f"Wrote: {overall_output_path}")

    all_areas_comparison_csv_path = output_dir / "all_areas_photovoltaic_comparison.csv"
    _write_rows_csv(all_areas_comparison_csv_path, all_areas_comparison_rows, comparison_headers)
    print(f"Wrote: {all_areas_comparison_csv_path}")

    _write_pv_hourly_normalized_shared_denominator_csvs_by_season(
        area_comparisons=area_comparisons,
        output_csv_prefix=hourly_normalized_shared_prefix,
        pv_code=pv_code,
        divide_by_area=True,
    )
    _write_pv_hourly_normalized_shared_denominator_csvs_by_season(
        area_comparisons=area_comparisons,
        output_csv_prefix=hourly_normalized_shared_no_area_prefix,
        pv_code=pv_code,
        divide_by_area=False,
    )


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
        default=Path("tools/comparison_outputs/Photovoltaic"),
        help="Directory where photovoltaic comparison outputs will be written.",
    )
    parser.add_argument(
        "--areas-config",
        type=Path,
        default=Path("tools/validation_areas_config.json"),
        help="Optional areas config to improve area -> scenario matching.",
    )
    parser.add_argument(
        "--pv-code",
        type=str,
        default="PV1",
        help="PV code used by CEA output filenames (e.g., PV1 -> PV_PV1_total_buildings.csv).",
    )
    parser.add_argument(
        "--hourly-normalized-shared-prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix for PV hourly normalized shared-denominator CSVs. "
            "Writes <prefix>_winter.csv, <prefix>_summer.csv, <prefix>_midseason.csv, "
            "and <prefix>_daily_error_summary.csv. "
            "Defaults to <output-dir>/all_areas_pv_hourly_normalized_shared_denominator."
        ),
    )
    parser.add_argument(
        "--hourly-normalized-shared-no-area-prefix",
        type=Path,
        default=None,
        help=(
            "Output prefix for PV hourly normalized shared-denominator CSVs without area division. "
            "Writes <prefix>_winter.csv, <prefix>_summer.csv, <prefix>_midseason.csv, "
            "and <prefix>_daily_error_summary.csv. "
            "Defaults to <output-dir>/all_areas_pv_hourly_normalized_shared_denominator_no_area."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    build_photovoltaic_comparison_excels(
        validation_root=args.validation_root,
        ucea_root=args.ucea_root,
        output_dir=args.output_dir,
        areas_config=args.areas_config,
        pv_code=args.pv_code,
        hourly_normalized_shared_prefix=args.hourly_normalized_shared_prefix,
        hourly_normalized_shared_no_area_prefix=args.hourly_normalized_shared_no_area_prefix,
    )


if __name__ == "__main__":
    main()
