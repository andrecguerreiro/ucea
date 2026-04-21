"""
Run two roof workflows for one CEA scenario and save comparable outputs.

Workflow 0:
    normal CEA flat roofs -> radiation -> photovoltaic

Workflow 1:
    geometry_generator custom roofs -> radiation -> photovoltaic
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import math
import time

import pandas as pd


WORKFLOW_0_NAME = "workflow0_normal_flat_roofs"
WORKFLOW_1_NAME = "workflow1_geometry_generator"
ROOF_REL_PATH = os.path.join("inputs", "building-geometry", "roof_surfaces.geojson")
TEMP_ROOF_SUFFIX = ".disabled_for_workflow0"
TEMP_PV_AZIMUTH_BACKUP_SUFFIX = ".pv_azimuth_convention_backup"

# Optional in-script defaults (edit these if you prefer running without CLI paths)
DEFAULT_SCENARIO = r"C:\Users\Andre\cea-scenarios\test-tilt"
DEFAULT_ROOF_FILE = r"C:\Users\Andre\cea-scenarios\test-tilt\inputs\building-geometry\roof_surfaces.geojson"
DEFAULT_COMPARISON_ROOT = r"C:\Users\Andre\cea-scenarios\test-tilt\outputs\data\roof-workflow-comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two roof workflows for one scenario and archive outputs for comparison."
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Path to CEA scenario. If omitted, DEFAULT_SCENARIO in this script is used.",
    )
    parser.add_argument(
        "--roof-file",
        default="",
        help=(
            "Custom roof file path. Workflow 1 reads only "
            "<scenario>/inputs/building-geometry/roof_surfaces.geojson."
        ),
    )
    parser.add_argument(
        "--comparison-root",
        default="",
        help=(
            "Folder to store workflow snapshots. If it already exists, it is overwritten. "
            "Default: <scenario>/outputs/data/roof-workflow-comparison."
        ),
    )
    parser.add_argument(
        "--skip-photovoltaic",
        action="store_true",
        help="Run radiation only and skip photovoltaic in all workflows.",
    )
    parser.add_argument(
        "--pv-panel",
        default="PV1",
        help="PV panel type for photovoltaic runs (default: PV1).",
    )
    parser.add_argument(
        "--include-insolation-feather",
        action="store_true",
        help="Also copy *_insolation_Whm2.feather files into snapshots (can be large).",
    )
    parser.add_argument(
        "--include-geometry-pickles",
        action="store_true",
        help=(
            "Also copy radiance geometry pickles into each workflow snapshot. "
            "Required for 3D geometry side-by-side visual comparison."
        ),
    )
    parser.add_argument(
        "--clean-first",
        action="store_true",
        help=(
            "Delete existing solar-radiation and potentials/solar outputs "
            "before each radiation run (workflow 0 and workflow 1)."
        ),
    )
    parser.add_argument(
        "--harmonise-pv-azimuth-convention",
        action="store_true",
        default=True,
        dest="harmonise_pv_azimuth_convention",
        help=(
            "Before each photovoltaic run, rotate sloped-roof metadata normals by 180° in XY so panel azimuth "
            "is in the same convention frame expected by the current PV AOI path. Metadata is restored afterwards. "
            "Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-harmonise-pv-azimuth-convention",
        action="store_false",
        dest="harmonise_pv_azimuth_convention",
        help=(
            "Disable temporary PV azimuth harmonisation and run photovoltaic with geometry metadata as generated "
            "by radiation."
        ),
    )
    return parser.parse_args()


def normalise(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def run_cea_script(script_name: str, scenario: str, extra_args: list[str] | None = None) -> None:
    command = [
        sys.executable,
        "-m",
        "cea.interfaces.cli.cli",
        script_name,
        "--scenario",
        scenario,
    ]
    if extra_args:
        command.extend(extra_args)
    print("[run] " + " ".join(command))
    subprocess.run(command, check=True)


def clean_outputs(scenario: str) -> None:
    targets = [
        os.path.join(scenario, "outputs", "data", "solar-radiation"),
        os.path.join(scenario, "outputs", "data", "potentials", "solar"),
    ]
    for target in targets:
        if os.path.isdir(target):
            _rmtree_with_retries(target)
            print(f"[clean] Removed: {target}")


def prepare_comparison_root(comparison_root: str) -> None:
    if os.path.isfile(comparison_root):
        raise FileExistsError(f"Comparison root points to a file, not a folder: {comparison_root}")
    if os.path.isdir(comparison_root):
        try:
            _rmtree_with_retries(comparison_root)
            print(f"[clean] Removed existing comparison root: {comparison_root}")
        except PermissionError:
            archived = f"{comparison_root}.locked_{int(time.time())}"
            os.replace(comparison_root, archived)
            print(f"[warn] Comparison root was locked and could not be deleted: {comparison_root}")
            print(f"[warn] Moved locked root to: {archived}")
    os.makedirs(comparison_root, exist_ok=True)


def _rmtree_with_retries(path: str, retries: int = 5, delay_seconds: float = 0.5) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


def ensure_no_stale_disabled_roof(roof_file: str) -> None:
    disabled_path = roof_file + TEMP_ROOF_SUFFIX
    if os.path.exists(disabled_path) and not os.path.exists(roof_file):
        os.replace(disabled_path, roof_file)
        print(f"[fix] Restored stale temporary roof file: {roof_file}")


def disable_roof_file_temporarily(roof_file: str) -> str:
    disabled_path = roof_file + TEMP_ROOF_SUFFIX
    if os.path.exists(disabled_path):
        raise FileExistsError(
            f"Cannot continue: temporary file already exists: {disabled_path}. "
            f"Restore or remove it first."
        )
    os.replace(roof_file, disabled_path)
    print(f"[step] Temporarily disabled roof file for flat-roof radiation: {roof_file}")
    return disabled_path


def restore_roof_file(roof_file: str, disabled_path: str) -> None:
    if os.path.exists(disabled_path):
        os.replace(disabled_path, roof_file)
        print(f"[step] Restored roof file: {roof_file}")


def copy_matching_files(source_folder: str, destination_folder: str, patterns: list[str]) -> None:
    if not os.path.isdir(source_folder):
        return

    os.makedirs(destination_folder, exist_ok=True)
    for pattern in patterns:
        for source_path in glob.glob(os.path.join(source_folder, pattern)):
            if os.path.isfile(source_path):
                destination_path = os.path.join(destination_folder, os.path.basename(source_path))
                shutil.copy2(source_path, destination_path)


def _list_geometry_csvs(scenario: str) -> list[str]:
    solar_radiation_folder = os.path.join(scenario, "outputs", "data", "solar-radiation")
    return sorted(glob.glob(os.path.join(solar_radiation_folder, "*_geometry.csv")))


def apply_temp_pv_azimuth_convention_harmonisation(scenario: str) -> list[tuple[str, str]]:
    geometry_paths = _list_geometry_csvs(scenario)
    backups: list[tuple[str, str]] = []
    if not geometry_paths:
        print("[info] No geometry CSV files found for temporary PV azimuth convention harmonisation.")
        return backups

    changed_rows = 0
    changed_files = 0
    for path in geometry_paths:
        backup_path = path + TEMP_PV_AZIMUTH_BACKUP_SUFFIX
        if os.path.exists(backup_path):
            raise FileExistsError(
                f"Temporary PV azimuth backup already exists: {backup_path}. "
                "Restore or remove backup files before running again."
            )
        shutil.copy2(path, backup_path)
        backups.append((path, backup_path))

        df = pd.read_csv(path)
        required = {"TYPE", "Xdir", "Ydir", "Zdir"}
        if not required.issubset(df.columns):
            continue

        roof_mask = df["TYPE"].astype(str).str.lower() == "roofs"
        zdir = pd.to_numeric(df["Zdir"], errors="coerce").clip(-1.0, 1.0)
        tilt_deg = zdir.apply(lambda z: math.degrees(math.acos(z)) if pd.notna(z) else float("nan"))
        sloped_roof_mask = roof_mask & (tilt_deg >= 5.0)
        if not sloped_roof_mask.any():
            continue

        df.loc[sloped_roof_mask, "Xdir"] = -pd.to_numeric(df.loc[sloped_roof_mask, "Xdir"], errors="coerce")
        df.loc[sloped_roof_mask, "Ydir"] = -pd.to_numeric(df.loc[sloped_roof_mask, "Ydir"], errors="coerce")
        df.to_csv(path, index=False)
        changed_files += 1
        changed_rows += int(sloped_roof_mask.sum())

    print(
        f"[step] Applied temporary PV azimuth convention harmonisation: "
        f"{changed_rows} sloped-roof rows changed across {changed_files} geometry files."
    )
    return backups


def restore_geometry_metadata_from_backups(backups: list[tuple[str, str]]) -> None:
    restored = 0
    for original_path, backup_path in backups:
        if not os.path.exists(backup_path):
            continue
        shutil.move(backup_path, original_path)
        restored += 1
    if restored:
        print(f"[step] Restored original geometry metadata from {restored} temporary backup files.")


def snapshot_outputs(
    scenario: str,
    workflow_snapshot_root: str,
    include_insolation_feather: bool,
    include_geometry_pickles: bool,
) -> None:
    solar_radiation_source = os.path.join(scenario, "outputs", "data", "solar-radiation")
    solar_radiation_destination = os.path.join(workflow_snapshot_root, "solar-radiation")

    radiation_patterns = ["*_geometry.csv", "*_radiation.csv", "buidling_materials.csv"]
    if include_insolation_feather:
        radiation_patterns.append("*_insolation_Whm2.feather")
    copy_matching_files(solar_radiation_source, solar_radiation_destination, radiation_patterns)

    solar_potential_source = os.path.join(scenario, "outputs", "data", "potentials", "solar")
    solar_potential_destination = os.path.join(workflow_snapshot_root, "potentials", "solar")
    if os.path.isdir(solar_potential_source):
        shutil.copytree(solar_potential_source, solar_potential_destination, dirs_exist_ok=True)

    if include_geometry_pickles:
        geometry_pickle_source = os.path.join(solar_radiation_source, "radiance_geometry_pickle")
        geometry_pickle_destination = os.path.join(workflow_snapshot_root, "radiance_geometry_pickle")
        if os.path.isdir(geometry_pickle_source):
            shutil.copytree(geometry_pickle_source, geometry_pickle_destination, dirs_exist_ok=True)
        else:
            print(f"[warn] Geometry pickle folder not found: {geometry_pickle_source}")


def run_workflow_1(
    scenario: str,
    workflow_snapshot_root: str,
    run_photovoltaic: bool,
    pv_panel: str,
    include_insolation_feather: bool,
    include_geometry_pickles: bool,
    clean_first: bool,
    harmonise_pv_azimuth_convention: bool,
) -> None:
    print(f"\n=== {WORKFLOW_1_NAME} ===")
    if clean_first:
        clean_outputs(scenario)

    run_cea_script("radiation", scenario, ["--buildings", ""])
    if run_photovoltaic:
        backups: list[tuple[str, str]] = []
        try:
            if harmonise_pv_azimuth_convention:
                backups = apply_temp_pv_azimuth_convention_harmonisation(scenario)
            run_cea_script(
                "photovoltaic",
                scenario,
                ["--buildings", "", "--panel-on-wall", "false", "--type-pvpanel", pv_panel],
            )
        finally:
            if backups:
                restore_geometry_metadata_from_backups(backups)

    snapshot_outputs(scenario, workflow_snapshot_root, include_insolation_feather, include_geometry_pickles)
    print(f"[done] Snapshot saved: {workflow_snapshot_root}")


def run_workflow_0(
    scenario: str,
    roof_file: str,
    workflow_snapshot_root: str,
    run_photovoltaic: bool,
    pv_panel: str,
    include_insolation_feather: bool,
    include_geometry_pickles: bool,
    clean_first: bool,
    harmonise_pv_azimuth_convention: bool,
) -> None:
    print(f"\n=== {WORKFLOW_0_NAME} ===")
    if clean_first:
        clean_outputs(scenario)

    disabled_path = disable_roof_file_temporarily(roof_file)
    try:
        run_cea_script("radiation", scenario, ["--buildings", ""])
    finally:
        restore_roof_file(roof_file, disabled_path)

    if run_photovoltaic:
        backups: list[tuple[str, str]] = []
        try:
            if harmonise_pv_azimuth_convention:
                backups = apply_temp_pv_azimuth_convention_harmonisation(scenario)
            run_cea_script(
                "photovoltaic",
                scenario,
                ["--buildings", "", "--panel-on-wall", "false", "--type-pvpanel", pv_panel],
            )
        finally:
            if backups:
                restore_geometry_metadata_from_backups(backups)

    snapshot_outputs(scenario, workflow_snapshot_root, include_insolation_feather, include_geometry_pickles)
    print(f"[done] Snapshot saved: {workflow_snapshot_root}")


def main() -> None:
    args = parse_args()

    scenario_input = args.scenario or DEFAULT_SCENARIO
    if not scenario_input:
        raise ValueError("No scenario path provided. Set --scenario or DEFAULT_SCENARIO in this script.")
    scenario = os.path.abspath(scenario_input)
    if not os.path.isdir(scenario):
        raise NotADirectoryError(f"Scenario does not exist: {scenario}")

    roof_file_default = os.path.join(scenario, ROOF_REL_PATH)
    roof_file_input = args.roof_file or DEFAULT_ROOF_FILE
    roof_file = os.path.abspath(roof_file_input) if roof_file_input else roof_file_default

    if normalise(roof_file) != normalise(roof_file_default):
        raise ValueError(
            "Workflow 1 only reads roof_surfaces.geojson from "
            "<scenario>/inputs/building-geometry/roof_surfaces.geojson. "
            "Use that location for this helper."
        )
    if not os.path.exists(roof_file):
        raise FileNotFoundError(f"Custom roof file not found: {roof_file}")

    ensure_no_stale_disabled_roof(roof_file)

    comparison_root_input = args.comparison_root or DEFAULT_COMPARISON_ROOT
    comparison_root = (
        os.path.abspath(comparison_root_input)
        if comparison_root_input
        else os.path.join(scenario, "outputs", "data", "roof-workflow-comparison")
    )
    prepare_comparison_root(comparison_root)

    run_photovoltaic = not args.skip_photovoltaic
    workflow_0_snapshot = os.path.join(comparison_root, WORKFLOW_0_NAME)
    workflow_1_snapshot = os.path.join(comparison_root, WORKFLOW_1_NAME)
    print("Two-workflow run settings")
    print(f"  scenario: {scenario}")
    print(f"  roof file: {roof_file}")
    print(f"  comparison root: {comparison_root}")
    print(f"  run photovoltaic: {run_photovoltaic}")
    print(f"  pv panel: {args.pv_panel}")
    print(f"  include insolation feather: {args.include_insolation_feather}")
    print(f"  include geometry pickles: {args.include_geometry_pickles}")
    print(f"  clean first: {args.clean_first}")
    print(f"  harmonise pv azimuth convention: {args.harmonise_pv_azimuth_convention}")

    run_workflow_1(
        scenario=scenario,
        workflow_snapshot_root=workflow_1_snapshot,
        run_photovoltaic=run_photovoltaic,
        pv_panel=args.pv_panel,
        include_insolation_feather=args.include_insolation_feather,
        include_geometry_pickles=args.include_geometry_pickles,
        clean_first=args.clean_first,
        harmonise_pv_azimuth_convention=args.harmonise_pv_azimuth_convention,
    )
    run_workflow_0(
        scenario=scenario,
        roof_file=roof_file,
        workflow_snapshot_root=workflow_0_snapshot,
        run_photovoltaic=run_photovoltaic,
        pv_panel=args.pv_panel,
        include_insolation_feather=args.include_insolation_feather,
        include_geometry_pickles=args.include_geometry_pickles,
        clean_first=args.clean_first,
        harmonise_pv_azimuth_convention=args.harmonise_pv_azimuth_convention,
    )
    print("\nAll workflows completed.")
    print(f"Compare outputs under: {comparison_root}")


if __name__ == "__main__":
    main()
