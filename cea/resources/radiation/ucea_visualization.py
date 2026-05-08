"""
Run a visualization-focused UX workflow for roof-comparison experiments.

This is intentionally aligned with `ucea.py` stage flow, but always runs the
two-workflow comparison path and launches the side-by-side viewer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cea.config import Configuration
from cea.inputlocator import InputLocator
from cea.resources.radiation import ucea as base


@dataclass
class UceaVisualizationRuntime:
    scenario: str
    comparison_root: str
    building: str
    metrics_output_dir: str
    images_output_dir: str
    viewer_started: bool


def _log(message: str) -> None:
    print(f"[ucea_visualization] {message}")


def _run_experiments(config: Configuration, scenario: str) -> UceaVisualizationRuntime:
    comparison_root = base._resolve_comparison_root(config, scenario)
    roof_file = base._require_roof_file(scenario)
    pv_panel = config.ucea.pv_panel

    base._run_python_module(
        "cea.resources.radiation.workflow_comparison",
        [
            "--scenario",
            scenario,
            "--roof-file",
            roof_file,
            "--comparison-root",
            comparison_root,
            "--include-geometry-pickles",
            "--clean-first",
            "--pv-panel",
            pv_panel,
            "--harmonise-pv-azimuth-convention",
        ],
    )

    base._run_python_module(
        "cea.resources.radiation.workflow_metrics_report",
        [
            "--comparison-root",
            comparison_root,
            "--level",
            "both",
            "--pv-panels",
            pv_panel,
        ],
    )
    metrics_output_dir = comparison_root

    locator = InputLocator(scenario)
    building = base._select_building_for_3d(locator, config.ucea.building_3d)
    images_output_dir = os.path.join(comparison_root, "workflow_comparison_3d_images")

    viewer_started = False
    try:
        base._run_python_module(
            "cea.resources.radiation.workflow_comparison_building_viewer",
            [
                "--comparison-root",
                comparison_root,
                "--building",
                building,
                "--export-images-dir",
                images_output_dir,
                "--show-sensors",
            ],
        )
        viewer_started = True
    except Exception as exc:
        _log(f"Could not launch interactive workflow-comparison building viewer: {exc}")

    return UceaVisualizationRuntime(
        scenario=scenario,
        comparison_root=comparison_root,
        building=building,
        metrics_output_dir=metrics_output_dir,
        images_output_dir=images_output_dir,
        viewer_started=viewer_started,
    )


def main(config: Configuration) -> None:
    scenario = base._resolve_scenario(config)
    os.makedirs(scenario, exist_ok=True)
    _log(f"Initialising visualization workflow for scenario: {scenario}")

    base._run_stage("Open geojson.io", lambda: base._open_geojson_io(config))
    _log(
        "Draw your polygon in geojson.io, then copy the GeoJSON text from the right panel. "
        "The runner will first try clipboard input and then ask for pasted text if needed."
    )

    coordinates = base._run_stage(
        "Polygon coordinate capture",
        lambda: base._capture_polygon_coordinates(config.ucea.polygon_timeout_minutes),
    )
    base._run_stage("Create site polygon", lambda: base._prepare_site_polygon(config, coordinates))
    base._run_stage("Scenario preparation scripts", lambda: base._run_data_preparation(config))
    base._run_stage(
        "Generate roof surfaces via fixedboxtrick",
        lambda: base._run_fixedboxtrick(scenario, coordinates),
    )

    runtime = base._run_stage(
        "Workflow comparison and side-by-side visualisation",
        lambda: _run_experiments(config, scenario),
    )

    _log("Run completed successfully.")
    _log(f"Comparison output root: {runtime.comparison_root}")
    _log(f"Scenario metrics: {os.path.join(runtime.metrics_output_dir, 'scenario_metrics.csv')}")
    _log(f"Building metrics: {os.path.join(runtime.metrics_output_dir, 'building_metrics.csv')}")
    _log(f"Side-by-side image export folder: {runtime.images_output_dir}")
    _log(f"Initial building used in viewer: {runtime.building}")
    if runtime.viewer_started:
        _log("Interactive workflow-comparison building viewer was launched.")


if __name__ == "__main__":
    main(Configuration())
