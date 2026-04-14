"""
Shared building-height enrichment helpers for zone and surroundings geometry.
"""

from __future__ import annotations

import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

HEIGHT_COLUMN = "altura(m)"
DEFAULT_FALLBACK_DISTANCE_M = 20.0
DEFAULT_MEDIAN_RADIUS_M = 50.0


def _metric_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        return None
    if hasattr(gdf.crs, "is_projected") and gdf.crs.is_projected:
        return gdf.crs
    try:
        estimated = gdf.estimate_utm_crs()
        if estimated is not None:
            return estimated
    except Exception:
        pass
    return "EPSG:3857"


def _read_height_points_subset(
    building_height_gpkg: str,
    buildings: gpd.GeoDataFrame,
    margin_m: float,
) -> gpd.GeoDataFrame:
    sample = gpd.read_file(building_height_gpkg, rows=1)
    if sample.empty:
        return gpd.GeoDataFrame(columns=[HEIGHT_COLUMN, "geometry"], geometry="geometry", crs=buildings.crs)
    if HEIGHT_COLUMN not in sample.columns:
        raise ValueError(
            f"Expected `{HEIGHT_COLUMN}` in `{building_height_gpkg}`, but the column was not found."
        )

    points_crs = sample.crs if sample.crs is not None else buildings.crs
    if points_crs is None:
        return gpd.GeoDataFrame(columns=[HEIGHT_COLUMN, "geometry"], geometry="geometry", crs=buildings.crs)

    buildings_for_bbox = buildings.to_crs(points_crs)
    minx, miny, maxx, maxy = buildings_for_bbox.total_bounds

    points = gpd.read_file(
        building_height_gpkg,
        bbox=(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m),
    )
    if HEIGHT_COLUMN not in points.columns:
        raise ValueError(
            f"Expected `{HEIGHT_COLUMN}` in `{building_height_gpkg}`, but the column was not found."
        )
    points[HEIGHT_COLUMN] = pd.to_numeric(points[HEIGHT_COLUMN], errors="coerce")
    points = points[points.geometry.notnull() & points[HEIGHT_COLUMN].notna()].copy()
    return points


def _apply_floor_validity_guard(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # CEA requires each building to have at least 1 m average floor height.
    heights = pd.to_numeric(buildings["height_ag"], errors="coerce")
    floors = pd.to_numeric(buildings["floors_ag"], errors="coerce").fillna(1)
    floors = floors.clip(lower=1)
    max_floors_by_height = np.floor(heights.fillna(1)).clip(lower=1)
    buildings["floors_ag"] = np.minimum(floors, max_floors_by_height).astype(int)
    return buildings


def apply_surroundings_floor_validity_guard(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Enforce surroundings floor-height validity to satisfy the strict CEA check
    requiring height_ag / floors_ag > 1.0.
    """
    guarded = buildings.copy()
    heights = pd.to_numeric(guarded["height_ag"], errors="coerce")
    floors = pd.to_numeric(guarded["floors_ag"], errors="coerce").fillna(1).clip(lower=1)
    strict_max_floors = np.floor(heights.fillna(1) - 1e-6).clip(lower=1)
    guarded["floors_ag"] = np.minimum(floors, strict_max_floors).astype(int)
    return guarded


def enrich_building_heights_from_points(
    buildings: gpd.GeoDataFrame,
    height_points: gpd.GeoDataFrame,
    fallback_max_distance_m: float = DEFAULT_FALLBACK_DISTANCE_M,
    reference_column: str = "reference",
) -> gpd.GeoDataFrame:
    if buildings.empty or height_points.empty:
        return buildings.copy()

    if fallback_max_distance_m <= 0:
        raise ValueError("`fallback_max_distance_m` must be greater than zero.")

    enriched = buildings.copy()
    metric_crs = _metric_crs(enriched)
    if metric_crs is None:
        print("Skipping INE height enrichment because geometry has no CRS.")
        return _apply_floor_validity_guard(enriched)

    buildings_metric = enriched.to_crs(metric_crs)
    points_metric = height_points.to_crs(metric_crs)

    selected_heights = {}
    selected_references = {}
    direct_matches = 0
    two_nearest_matches = 0
    radius_median_matches = 0

    # Stage 1: direct matches from points inside building footprints.
    # Stage 2: 2-nearest average fallback.
    # Stage 3: if stage 2 fails, use INE median within a predefined radius around the building.
    for building_index, building_geometry in buildings_metric.geometry.items():
        inside = points_metric[points_metric.within(building_geometry)]
        if not inside.empty:
            if len(inside) == 1:
                selected_heights[building_index] = float(inside[HEIGHT_COLUMN].iloc[0])
            else:
                centroid = building_geometry.centroid
                closest_point_index = inside.distance(centroid).idxmin()
                selected_heights[building_index] = float(points_metric.loc[closest_point_index, HEIGHT_COLUMN])
            selected_references[building_index] = "INE"
            direct_matches += 1
            continue

        distances = points_metric.distance(building_geometry)
        nearest_two = distances.nsmallest(2)
        if len(nearest_two) >= 2:
            second_nearest_distance = float(nearest_two.iloc[1])
            if second_nearest_distance <= fallback_max_distance_m:
                first_height = float(points_metric.loc[nearest_two.index[0], HEIGHT_COLUMN])
                second_height = float(points_metric.loc[nearest_two.index[1], HEIGHT_COLUMN])
                selected_heights[building_index] = (first_height + second_height) / 2.0
                selected_references[building_index] = "INE Assumption"
                two_nearest_matches += 1
                continue

        nearby_points = points_metric.loc[distances <= DEFAULT_MEDIAN_RADIUS_M, HEIGHT_COLUMN]
        if not nearby_points.empty:
            selected_heights[building_index] = float(np.median(nearby_points.to_numpy(dtype=float)))
            selected_references[building_index] = "INE Assumption"
            radius_median_matches += 1

    if selected_heights:
        selected_height_series = pd.Series(selected_heights, dtype=float)
        enriched.loc[selected_height_series.index, "height_ag"] = selected_height_series
        if reference_column not in enriched.columns:
            enriched[reference_column] = ""
        selected_ref_series = pd.Series(selected_references, dtype=str)
        enriched.loc[selected_ref_series.index, reference_column] = selected_ref_series

    unchanged = len(enriched) - direct_matches - two_nearest_matches - radius_median_matches
    print(
        f"INE height enrichment applied: {direct_matches} direct matches, "
        f"{two_nearest_matches} 2-nearest matches, "
        f"{radius_median_matches} radius-median matches (<= {DEFAULT_MEDIAN_RADIUS_M:.0f} m), "
        f"{unchanged} unchanged."
    )
    return _apply_floor_validity_guard(enriched)


def enrich_building_heights_from_gpkg(
    buildings: gpd.GeoDataFrame,
    building_height_gpkg: Optional[str],
    fallback_max_distance_m: float = DEFAULT_FALLBACK_DISTANCE_M,
    reference_column: str = "reference",
) -> gpd.GeoDataFrame:
    if not building_height_gpkg:
        return buildings.copy()

    gpkg_path = os.path.abspath(os.path.expanduser(str(building_height_gpkg)))
    if not os.path.exists(gpkg_path):
        print(
            f"INE height GeoPackage not found at `{gpkg_path}`. "
            "Keeping existing building heights."
        )
        return buildings.copy()

    if fallback_max_distance_m <= 0:
        raise ValueError("`fallback_max_distance_m` must be greater than zero.")

    search_margin_m = max(fallback_max_distance_m, DEFAULT_MEDIAN_RADIUS_M, 1.0)
    height_points = _read_height_points_subset(gpkg_path, buildings, search_margin_m)
    if height_points.empty:
        print("No INE height points were found near the selected area. Keeping existing building heights.")
        return buildings.copy()

    return enrich_building_heights_from_points(
        buildings=buildings,
        height_points=height_points,
        fallback_max_distance_m=fallback_max_distance_m,
        reference_column=reference_column,
    )
