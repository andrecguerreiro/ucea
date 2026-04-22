"""
Interactive side-by-side building viewer for workflow0/workflow1 snapshots.

Displays the same building in two panels:
- left: workflow0_normal_flat_roofs
- right: workflow1_geometry_generator
"""

from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from py4design.py3dmodel.fetch import points_frm_occface

from cea.resources.radiation.geometry_generator import BuildingGeometry


WORKFLOW_0 = "workflow0_normal_flat_roofs"
WORKFLOW_1 = "workflow1_geometry_generator"
WORKFLOWS = [WORKFLOW_0, WORKFLOW_1]

DEFAULT_SCENARIO = r"C:\Users\Andre\cea-scenarios\test-case"
DEFAULT_COMPARISON_ROOT = r"C:\Users\Andre\cea-scenarios\test-case\outputs\data\roof-workflow-comparison"
DEFAULT_SHOW_SENSORS = True
DEFAULT_MAX_SENSORS = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an interactive side-by-side viewer for workflow0/workflow1 building geometry."
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Path to CEA scenario. Used for default comparison-root if omitted.",
    )
    parser.add_argument(
        "--comparison-root",
        default="",
        help=(
            "Path to workflow comparison root containing workflow0/workflow1 snapshots. "
            "Default: <scenario>/outputs/data/roof-workflow-comparison."
        ),
    )
    parser.add_argument(
        "--building",
        default="",
        help="Optional initial building to load. Defaults to first common building.",
    )
    parser.add_argument(
        "--export-images-dir",
        default="",
        help="Optional output folder to export one side-by-side PNG per building.",
    )
    parser.add_argument(
        "--show-sensors",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW_SENSORS,
        help="Show roof sensor locations (use --no-show-sensors to disable).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip interactive GUI and run export-only mode.",
    )
    parser.add_argument(
        "--max-sensors",
        type=int,
        default=DEFAULT_MAX_SENSORS,
        help=(
            "Maximum number of roof sensors to plot per panel (deterministic downsampling). "
            f"Default: {DEFAULT_MAX_SENSORS}."
        ),
    )
    return parser.parse_args()


def resolve_comparison_root(args: argparse.Namespace) -> str:
    if args.comparison_root.strip():
        return os.path.abspath(args.comparison_root.strip())
    if args.scenario.strip():
        return os.path.join(
            os.path.abspath(args.scenario.strip()),
            "outputs",
            "data",
            "roof-workflow-comparison",
        )
    return DEFAULT_COMPARISON_ROOT


def workflow_zone_pickle_dir(comparison_root: str, workflow: str) -> str:
    return os.path.join(comparison_root, workflow, "radiance_geometry_pickle", "zone")


def workflow_metadata_dir(comparison_root: str, workflow: str) -> str:
    return os.path.join(comparison_root, workflow, "solar-radiation")


def list_buildings_for_workflow(zone_pickle_dir: str) -> set[str]:
    if not os.path.isdir(zone_pickle_dir):
        return set()
    buildings = set()
    for name in os.listdir(zone_pickle_dir):
        path = os.path.join(zone_pickle_dir, name)
        if os.path.isfile(path):
            buildings.add(name)
    return buildings


def list_common_buildings(comparison_root: str) -> list[str]:
    wf0 = list_buildings_for_workflow(workflow_zone_pickle_dir(comparison_root, WORKFLOW_0))
    wf1 = list_buildings_for_workflow(workflow_zone_pickle_dir(comparison_root, WORKFLOW_1))
    common = sorted(wf0.intersection(wf1))
    if not common:
        raise FileNotFoundError(
            "No common building geometry pickles found between workflow0 and workflow1. "
            "Run workflow comparison with --include-geometry-pickles first."
        )
    return common


def face_to_points_xyz(face: Any) -> np.ndarray | None:
    if isinstance(face, np.ndarray):
        points = np.asarray(face, dtype=float)
    elif isinstance(face, (list, tuple)) and len(face) > 0:
        points = np.asarray(face, dtype=float)
    else:
        points = np.asarray(points_frm_occface(face), dtype=float)

    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        return None
    return points[:, :3]


def occ_faces_to_polygons_xyz(faces: list[Any] | None) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not faces:
        return polygons
    for face in faces:
        points = face_to_points_xyz(face)
        if points is not None:
            polygons.append(points)
    return polygons


def add_surfaces(ax, polygons: list[np.ndarray], color: str, alpha: float) -> None:
    if not polygons:
        return
    collection = Poly3DCollection(
        polygons, facecolor=color, edgecolor="black", linewidth=0.4, alpha=alpha
    )
    ax.add_collection3d(collection)


def downsample_dataframe_deterministic(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if max_rows <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=max_rows, replace=False))
    return df.iloc[idx].copy()


def load_roof_sensor_points(metadata_dir: str, building: str, max_sensors: int) -> np.ndarray:
    metadata_path = os.path.join(metadata_dir, f"{building}_geometry.csv")
    if not os.path.exists(metadata_path):
        return np.empty((0, 3), dtype=float)
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception:
        return np.empty((0, 3), dtype=float)

    required = {"TYPE", "Xcoor", "Ycoor", "Zcoor"}
    if not required.issubset(metadata.columns):
        return np.empty((0, 3), dtype=float)

    roofs = metadata[metadata["TYPE"].astype(str).str.lower() == "roofs"].copy()
    if roofs.empty:
        return np.empty((0, 3), dtype=float)

    roofs = roofs[["Xcoor", "Ycoor", "Zcoor"]].apply(pd.to_numeric, errors="coerce").dropna()
    roofs = downsample_dataframe_deterministic(roofs, max_sensors)
    if roofs.empty:
        return np.empty((0, 3), dtype=float)
    return roofs[["Xcoor", "Ycoor", "Zcoor"]].to_numpy(dtype=float)


def load_building_geometry(zone_pickle_dir: str, building: str) -> BuildingGeometry:
    path = os.path.join(zone_pickle_dir, building)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Building geometry pickle not found: {path}")
    return BuildingGeometry.load(path)


def collect_bounds(points_sets: list[np.ndarray]) -> dict[str, tuple[float, float] | float]:
    valid_sets = [points for points in points_sets if len(points)]
    if not valid_sets:
        return {
            "xlim": (0.0, 1.0),
            "ylim": (0.0, 1.0),
            "zlim": (0.0, 1.0),
            "sensor_size": 6.0,
        }

    xyz = np.vstack(valid_sets)
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    centre = 0.5 * (min_xyz + max_xyz)
    span = float(np.max(max_xyz - min_xyz))
    if span <= 0:
        span = 1.0
    half = 0.6 * span
    return {
        "xlim": (float(centre[0] - half), float(centre[0] + half)),
        "ylim": (float(centre[1] - half), float(centre[1] + half)),
        "zlim": (float(centre[2] - half), float(centre[2] + half)),
        "sensor_size": max(2.0, 0.006 * span),
    }


def render_workflow_panel(
    ax,
    *,
    comparison_root: str,
    workflow: str,
    building: str,
    show_sensors: bool,
    max_sensors: int,
    bounds: dict[str, tuple[float, float] | float],
) -> int:
    zone_pickle_dir = workflow_zone_pickle_dir(comparison_root, workflow)
    metadata_dir = workflow_metadata_dir(comparison_root, workflow)
    geometry = load_building_geometry(zone_pickle_dir, building)

    walls = occ_faces_to_polygons_xyz(getattr(geometry, "walls", []))
    roofs = occ_faces_to_polygons_xyz(getattr(geometry, "roofs", []))
    windows = occ_faces_to_polygons_xyz(getattr(geometry, "windows", []))
    sensor_points = (
        load_roof_sensor_points(metadata_dir, building, max_sensors=max_sensors)
        if show_sensors
        else np.empty((0, 3), dtype=float)
    )

    ax.clear()
    add_surfaces(ax, walls, color="lightgrey", alpha=0.25)
    add_surfaces(ax, roofs, color="steelblue", alpha=0.5)
    add_surfaces(ax, windows, color="gold", alpha=0.35)

    if len(sensor_points):
        ax.scatter(
            sensor_points[:, 0],
            sensor_points[:, 1],
            sensor_points[:, 2],
            s=float(bounds["sensor_size"]),
            c="crimson",
            alpha=0.7,
            depthshade=False,
        )

    ax.set_xlim(*bounds["xlim"])
    ax.set_ylim(*bounds["ylim"])
    ax.set_zlim(*bounds["zlim"])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.view_init(elev=24, azim=-60)
    title = "Workflow 0 - Flat roofs" if workflow == WORKFLOW_0 else "Workflow 1 - Custom roofs"
    ax.set_title(title)
    return int(len(sensor_points))


def compute_bounds_for_building(
    comparison_root: str,
    building: str,
    show_sensors: bool,
    max_sensors: int,
) -> dict[str, tuple[float, float] | float]:
    points_sets: list[np.ndarray] = []
    for workflow in WORKFLOWS:
        zone_pickle_dir = workflow_zone_pickle_dir(comparison_root, workflow)
        geometry = load_building_geometry(zone_pickle_dir, building)
        for surface_name in ["walls", "roofs", "windows"]:
            polygons = occ_faces_to_polygons_xyz(getattr(geometry, surface_name, []))
            if polygons:
                points_sets.append(np.vstack(polygons))
        if show_sensors:
            metadata_dir = workflow_metadata_dir(comparison_root, workflow)
            sensor_points = load_roof_sensor_points(metadata_dir, building, max_sensors=max_sensors)
            if len(sensor_points):
                points_sets.append(sensor_points)
    return collect_bounds(points_sets)


def render_side_by_side(
    figure: Figure,
    ax_left,
    ax_right,
    *,
    comparison_root: str,
    building: str,
    show_sensors: bool,
    max_sensors: int,
) -> tuple[int, int]:
    bounds = compute_bounds_for_building(
        comparison_root=comparison_root,
        building=building,
        show_sensors=show_sensors,
        max_sensors=max_sensors,
    )
    left_count = render_workflow_panel(
        ax_left,
        comparison_root=comparison_root,
        workflow=WORKFLOW_0,
        building=building,
        show_sensors=show_sensors,
        max_sensors=max_sensors,
        bounds=bounds,
    )
    right_count = render_workflow_panel(
        ax_right,
        comparison_root=comparison_root,
        workflow=WORKFLOW_1,
        building=building,
        show_sensors=show_sensors,
        max_sensors=max_sensors,
        bounds=bounds,
    )
    figure.suptitle(f"Roof workflow comparison - {building}", fontsize=14)
    return left_count, right_count


def export_building_images(
    comparison_root: str,
    export_images_dir: str,
    *,
    show_sensors: bool,
    max_sensors: int,
) -> list[str]:
    buildings = list_common_buildings(comparison_root)
    os.makedirs(export_images_dir, exist_ok=True)

    exported_paths: list[str] = []
    for building in buildings:
        figure = Figure(figsize=(14, 7), dpi=100)
        ax_left = figure.add_subplot(1, 2, 1, projection="3d")
        ax_right = figure.add_subplot(1, 2, 2, projection="3d")
        left_count, right_count = render_side_by_side(
            figure,
            ax_left,
            ax_right,
            comparison_root=comparison_root,
            building=building,
            show_sensors=show_sensors,
            max_sensors=max_sensors,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = os.path.join(export_images_dir, f"{building}_workflow_comparison_3d.png")
        figure.savefig(out_path, dpi=220)
        plt.close(figure)
        exported_paths.append(out_path)
        print(
            f"Exported image: {out_path} "
            f"(roof sensors plotted: wf0={left_count}, wf1={right_count})"
        )
    return exported_paths


def open_viewer(
    comparison_root: str,
    *,
    initial_building: str,
    show_sensors_default: bool,
    max_sensors: int,
) -> None:
    buildings = list_common_buildings(comparison_root)
    default_building = initial_building if initial_building in buildings else buildings[0]

    root = tk.Tk()
    root.title("CEA Roof Workflow Side-by-Side Viewer")
    root.geometry("1450x900")

    controls = ttk.Frame(root, padding=8)
    controls.pack(side=tk.TOP, fill=tk.X)

    ttk.Label(controls, text="Building:").pack(side=tk.LEFT, padx=(0, 8))
    selected_building = tk.StringVar(value=default_building)
    show_sensors_var = tk.BooleanVar(value=show_sensors_default)

    combo = ttk.Combobox(
        controls, textvariable=selected_building, values=buildings, state="readonly", width=30
    )
    combo.pack(side=tk.LEFT)

    sensors_check = ttk.Checkbutton(
        controls,
        text="Show roof sensors",
        variable=show_sensors_var,
    )
    sensors_check.pack(side=tk.LEFT, padx=(12, 0))

    figure = Figure(figsize=(14, 7), dpi=100)
    ax_left = figure.add_subplot(1, 2, 1, projection="3d")
    ax_right = figure.add_subplot(1, 2, 2, projection="3d")
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    status_text = tk.StringVar(value=f"Comparison root: {comparison_root}")
    ttk.Label(root, textvariable=status_text, padding=(8, 4)).pack(side=tk.BOTTOM, fill=tk.X)

    def refresh(*_):
        building = selected_building.get().strip()
        try:
            left_count, right_count = render_side_by_side(
                figure,
                ax_left,
                ax_right,
                comparison_root=comparison_root,
                building=building,
                show_sensors=bool(show_sensors_var.get()),
                max_sensors=max_sensors,
            )
            status_text.set(
                f"Loaded {building} | show sensors: {bool(show_sensors_var.get())} | "
                f"wf0 sensors: {left_count} | wf1 sensors: {right_count}"
            )
            figure.tight_layout(rect=(0, 0, 1, 0.95))
            canvas.draw_idle()
        except Exception as exc:
            status_text.set(f"Failed to load {building}: {exc}")

    combo.bind("<<ComboboxSelected>>", refresh)
    sensors_check.configure(command=refresh)
    refresh()
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.max_sensors < 1:
        raise ValueError("--max-sensors must be at least 1.")

    comparison_root = resolve_comparison_root(args)
    if not os.path.isdir(comparison_root):
        raise NotADirectoryError(f"Comparison folder does not exist: {comparison_root}")

    buildings = list_common_buildings(comparison_root)
    initial_building = args.building.strip() or buildings[0]
    if initial_building not in buildings:
        raise ValueError(
            f"Building '{initial_building}' is not available in both workflows. "
            f"Available: {', '.join(buildings[:20])}{' ...' if len(buildings) > 20 else ''}"
        )

    print(f"Comparison root: {comparison_root}")
    print(f"Initial building: {initial_building}")
    print(f"Show sensors: {args.show_sensors}")

    if args.export_images_dir.strip():
        export_dir = os.path.abspath(args.export_images_dir.strip())
        exported_paths = export_building_images(
            comparison_root=comparison_root,
            export_images_dir=export_dir,
            show_sensors=args.show_sensors,
            max_sensors=args.max_sensors,
        )
        print(f"Exported {len(exported_paths)} images to: {export_dir}")

    if args.no_gui:
        return

    open_viewer(
        comparison_root=comparison_root,
        initial_building=initial_building,
        show_sensors_default=args.show_sensors,
        max_sensors=args.max_sensors,
    )


if __name__ == "__main__":
    main()
