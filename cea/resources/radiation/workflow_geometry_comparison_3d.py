"""
Visualise side-by-side 3D building geometry for workflow0/workflow1.

Each workflow panel shows:
- Full selected-building shell surfaces (walls, roofs, windows)
- Roof normals from BuildingGeometry (face-level geometry normals)
- Roof normals from metadata CSV (sensor-level normals)
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from py4design.py3dmodel.fetch import points_frm_occface

from cea.resources.radiation.geometry_generator import BuildingGeometry


WORKFLOW_0 = "workflow0_normal_flat_roofs"
WORKFLOW_1 = "workflow1_geometry_generator"
WORKFLOWS = [WORKFLOW_0, WORKFLOW_1]

DEFAULT_SCENARIO = r"C:\Users\Andre\cea-scenarios\Validation\Alameda"
DEFAULT_COMPARISON_ROOT = r"C:\Users\Andre\cea-scenarios\Validation\Alameda\outputs\data\roof-workflow-comparison"
DEFAULT_OUTPUT_FIGURE = r"C:\Users\Andre\cea-scenarios\vis\outputs\data\roof-workflow-comparison\output.png"
DEFAULT_BUILDING = "B1065"
DEFAULT_MAX_METADATA_ARROWS = 300
DEFAULT_SHOW = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side 3D workflow geometry visualisation for one building "
            "using workflow snapshots under roof-workflow-comparison."
        )
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
            "Path to one workflow comparison folder containing workflow0/workflow1. "
            "Default: <scenario>/outputs/data/roof-workflow-comparison."
        ),
    )
    parser.add_argument(
        "--building",
        default=DEFAULT_BUILDING,
        help=f"Building name to visualise. Default: {DEFAULT_BUILDING}",
    )
    parser.add_argument(
        "--output-figure",
        default="",
        help=(
            "Output PNG path. Default: "
            "<comparison-root>/<building>_workflow_geometry_comparison_3d.png"
        ),
    )
    parser.add_argument(
        "--max-metadata-arrows",
        type=int,
        default=DEFAULT_MAX_METADATA_ARROWS,
        help=f"Maximum metadata roof-normal arrows per workflow panel. Default: {DEFAULT_MAX_METADATA_ARROWS}",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW,
        help="Show interactive figure window after saving PNG (use --no-show to disable).",
    )
    return parser.parse_args()


def workflow_paths(comparison_root: str, workflow: str, building: str) -> dict[str, str]:
    workflow_root = os.path.join(comparison_root, workflow)
    return {
        "workflow_root": workflow_root,
        "geometry_pickle": os.path.join(workflow_root, "radiance_geometry_pickle", "zone", building),
        "metadata_csv": os.path.join(workflow_root, "solar-radiation", f"{building}_geometry.csv"),
    }


def normalise_vectors(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0.0] = 1.0
    return vectors / norms[:, None]


def downsample_dataframe_deterministic(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if max_rows <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= max_rows:
        return df
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=max_rows, replace=False))
    return df.iloc[idx].copy()


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
    if not faces:
        return []

    polygons: list[np.ndarray] = []
    for face in faces:
        points = face_to_points_xyz(face)
        if points is None:
            continue
        polygons.append(points)
    return polygons


def polygon_centroid_xyz(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) >= 2 and np.allclose(points_xyz[0], points_xyz[-1]):
        return points_xyz[:-1].mean(axis=0)
    return points_xyz.mean(axis=0)


def prepare_geometry_data(building_geometry: BuildingGeometry) -> dict[str, np.ndarray | list[np.ndarray]]:
    walls = occ_faces_to_polygons_xyz(getattr(building_geometry, "walls", []))
    roofs = occ_faces_to_polygons_xyz(getattr(building_geometry, "roofs", []))
    windows = occ_faces_to_polygons_xyz(getattr(building_geometry, "windows", []))

    centroids = np.array([polygon_centroid_xyz(p) for p in roofs], dtype=float) if roofs else np.empty((0, 3), dtype=float)
    roof_normals = np.asarray(getattr(building_geometry, "normals_roofs", []), dtype=float)
    if roof_normals.ndim == 1 and len(roof_normals) > 0:
        roof_normals = roof_normals.reshape(1, -1)
    if roof_normals.ndim != 2 or roof_normals.shape[1] < 3:
        roof_normals = np.empty((0, 3), dtype=float)
    else:
        roof_normals = roof_normals[:, :3]

    n = min(len(centroids), len(roof_normals))
    if n > 0:
        centroids = centroids[:n]
        roof_normals = normalise_vectors(roof_normals[:n])
    else:
        centroids = np.empty((0, 3), dtype=float)
        roof_normals = np.empty((0, 3), dtype=float)

    return {
        "walls": walls,
        "roofs": roofs,
        "windows": windows,
        "roof_centroids": centroids,
        "roof_normals": roof_normals,
    }


def load_metadata_roof_normals(metadata_csv: str, max_metadata_arrows: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(metadata_csv)
    except Exception as exc:
        raise ValueError(f"Failed to read metadata CSV: {metadata_csv}. Error: {exc}") from exc

    required = {"TYPE", "Xcoor", "Ycoor", "Zcoor", "Xdir", "Ydir", "Zdir"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Metadata CSV missing required columns {missing}: {metadata_csv}")

    roof = df[df["TYPE"] == "roofs"].copy()
    if roof.empty:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    roof = roof[["Xcoor", "Ycoor", "Zcoor", "Xdir", "Ydir", "Zdir"]].apply(pd.to_numeric, errors="coerce").dropna()
    roof = downsample_dataframe_deterministic(roof, max_metadata_arrows)

    points = roof[["Xcoor", "Ycoor", "Zcoor"]].to_numpy(dtype=float)
    normals = roof[["Xdir", "Ydir", "Zdir"]].to_numpy(dtype=float)
    normals = normalise_vectors(normals)
    return points, normals


def load_workflow_data(
    comparison_root: str,
    workflow: str,
    building: str,
    max_metadata_arrows: int,
) -> dict[str, Any]:
    paths = workflow_paths(comparison_root, workflow, building)
    workflow_root = paths["workflow_root"]
    geometry_pickle = paths["geometry_pickle"]
    metadata_csv = paths["metadata_csv"]

    if not os.path.isdir(workflow_root):
        raise NotADirectoryError(f"Workflow folder does not exist: {workflow_root}")

    if not os.path.exists(geometry_pickle):
        raise FileNotFoundError(
            "Missing workflow geometry pickle for side-by-side geometry comparison: "
            f"{geometry_pickle}. Rerun workflow comparison with --include-geometry-pickles."
        )

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Missing workflow metadata CSV: {metadata_csv}")

    building_geometry = BuildingGeometry.load(geometry_pickle)
    geometry_data = prepare_geometry_data(building_geometry)
    metadata_points, metadata_normals = load_metadata_roof_normals(metadata_csv, max_metadata_arrows)

    return {
        "workflow": workflow,
        "geometry": geometry_data,
        "metadata_points": metadata_points,
        "metadata_normals": metadata_normals,
        "paths": paths,
    }


def collect_bounds(workflow_data: dict[str, dict[str, Any]]) -> dict[str, tuple[float, float] | float]:
    sets: list[np.ndarray] = []
    for data in workflow_data.values():
        geometry = data["geometry"]
        for key in ["walls", "roofs", "windows"]:
            polygons = geometry[key]
            if polygons:
                sets.append(np.vstack(polygons))
        if len(geometry["roof_centroids"]):
            sets.append(geometry["roof_centroids"])
        if len(data["metadata_points"]):
            sets.append(data["metadata_points"])

    if not sets:
        return {
            "xlim": (0.0, 1.0),
            "ylim": (0.0, 1.0),
            "zlim": (0.0, 1.0),
            "arrow_length": 1.0,
        }

    xyz = np.vstack(sets)
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
        "arrow_length": max(0.08 * span, 1.0),
    }


def add_surfaces(ax, polygons: list[np.ndarray], color: str, alpha: float) -> None:
    if not polygons:
        return
    collection = Poly3DCollection(polygons, facecolor=color, edgecolor="black", linewidth=0.5, alpha=alpha)
    ax.add_collection3d(collection)


def plot_workflow_panel(ax, data: dict[str, Any], bounds: dict[str, tuple[float, float] | float]) -> None:
    geometry = data["geometry"]
    add_surfaces(ax, geometry["walls"], color="lightgrey", alpha=0.25)
    add_surfaces(ax, geometry["roofs"], color="steelblue", alpha=0.45)
    add_surfaces(ax, geometry["windows"], color="gold", alpha=0.35)

    arrow_length = float(bounds["arrow_length"])
    roof_centroids = geometry["roof_centroids"]
    roof_normals = geometry["roof_normals"]
    if len(roof_centroids) and len(roof_normals):
        ax.quiver(
            roof_centroids[:, 0],
            roof_centroids[:, 1],
            roof_centroids[:, 2],
            roof_normals[:, 0],
            roof_normals[:, 1],
            roof_normals[:, 2],
            length=arrow_length,
            color="royalblue",
            normalize=True,
            linewidth=1.3,
        )

    metadata_points = data["metadata_points"]
    metadata_normals = data["metadata_normals"]
    if len(metadata_points) and len(metadata_normals):
        ax.quiver(
            metadata_points[:, 0],
            metadata_points[:, 1],
            metadata_points[:, 2],
            metadata_normals[:, 0],
            metadata_normals[:, 1],
            metadata_normals[:, 2],
            length=0.7 * arrow_length,
            color="crimson",
            normalize=True,
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_xlim(*bounds["xlim"])
    ax.set_ylim(*bounds["ylim"])
    ax.set_zlim(*bounds["zlim"])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect((1.0, 1.0, 0.8))
    ax.view_init(elev=24, azim=-60)


def create_figure(
    workflow_data: dict[str, dict[str, Any]],
    building: str,
    output_figure: str,
    show: bool,
) -> None:
    bounds = collect_bounds(workflow_data)
    figure_width = 7 * len(WORKFLOWS)
    fig = plt.figure(figsize=(figure_width, 7))
    axes = [fig.add_subplot(1, len(WORKFLOWS), i + 1, projection="3d") for i in range(len(WORKFLOWS))]

    titles = {
        WORKFLOW_0: "CEA Building + Roof",
        WORKFLOW_1: "CEA Building + OVEN Generated Roof",
    }

    for ax, workflow in zip(axes, WORKFLOWS):
        plot_workflow_panel(ax, workflow_data[workflow], bounds)
        ax.set_title(titles[workflow])

    legend_handles = [
        Line2D([0], [0], color="black", lw=8, alpha=0.25, label="Walls"),
        Line2D([0], [0], color="steelblue", lw=8, alpha=0.45, label="Roofs"),
        Line2D([0], [0], color="goldenrod", lw=8, alpha=0.35, label="Windows"),
        Line2D([0], [0], color="crimson", lw=2, label="Roof normals"),
    ]

    fig.suptitle(f"3D workflow geometry comparison - {building}", fontsize=14)
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))

    os.makedirs(os.path.dirname(os.path.abspath(output_figure)), exist_ok=True)
    fig.savefig(output_figure, dpi=220)
    print(f"Saved figure: {os.path.abspath(output_figure)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    scenario_input = args.scenario or DEFAULT_SCENARIO
    comparison_root_input = args.comparison_root or DEFAULT_COMPARISON_ROOT

    scenario = os.path.abspath(scenario_input) if scenario_input else ""
    if scenario and not os.path.isdir(scenario):
        raise NotADirectoryError(f"Scenario does not exist: {scenario}")

    if comparison_root_input:
        comparison_root = os.path.abspath(comparison_root_input)
    elif scenario:
        comparison_root = os.path.join(scenario, "outputs", "data", "roof-workflow-comparison")
    else:
        raise ValueError(
            "No comparison root provided. Set --comparison-root, or provide --scenario "
            "to use <scenario>/outputs/data/roof-workflow-comparison."
        )

    if not os.path.isdir(comparison_root):
        raise NotADirectoryError(f"Comparison folder does not exist: {comparison_root}")

    if args.output_figure or DEFAULT_OUTPUT_FIGURE:
        output_figure = os.path.abspath(args.output_figure or DEFAULT_OUTPUT_FIGURE)
    else:
        output_figure = os.path.join(
            comparison_root,
            f"{args.building}_workflow_geometry_comparison_3d.png",
        )

    return comparison_root, output_figure


def main() -> None:
    args = parse_args()
    if args.max_metadata_arrows < 1:
        raise ValueError("--max-metadata-arrows must be at least 1.")

    comparison_root, output_figure = resolve_paths(args)
    building = args.building.strip()
    if not building:
        raise ValueError("Building name cannot be empty.")

    workflow_data = {
        workflow: load_workflow_data(comparison_root, workflow, building, args.max_metadata_arrows)
        for workflow in WORKFLOWS
    }

    print("3D workflow geometry comparison settings")
    print(f"  comparison root: {comparison_root}")
    print(f"  building: {building}")
    print(f"  output figure: {output_figure}")
    print(f"  max metadata arrows: {args.max_metadata_arrows}")
    print(f"  show figure: {args.show}")

    create_figure(workflow_data, building, output_figure, show=args.show)
    print("Workflow geometry visualisation completed.")


if __name__ == "__main__":
    main()
