# Data Management Module

## Main API
- `zone_helper.zone_helper(locator, config) -> None` - Build `zone.shp` and typology inputs from site polygon + OSM.
- `surroundings_helper.geometry_extractor_osm(locator, config) -> None` - Build `surroundings.shp` from OSM around zone.
- `height_enrichment.enrich_building_heights_from_gpkg(buildings, building_height_gpkg, fallback_max_distance_m=20.0, reference_column="reference") -> GeoDataFrame` - Optionally enrich `height_ag`/`floors_ag` from INE point heights.
- `height_enrichment.enrich_building_heights_from_points(buildings, height_points, fallback_max_distance_m=20.0, reference_column="reference") -> GeoDataFrame` - In-memory variant used by tests and helper orchestration.
- `height_enrichment.apply_surroundings_floor_validity_guard(buildings) -> GeoDataFrame` - Adjust `floors_ag` for surroundings so `height_ag / floors_ag > 1.0`.

## Key Patterns
### DO: Keep INE enrichment optional and non-breaking
```python
shapefile = enrich_building_heights_from_gpkg(
    buildings=shapefile,
    building_height_gpkg=config.zone_helper.building_height_gpkg,
    fallback_max_distance_m=config.zone_helper.ine_fallback_max_distance_m,
    reference_column="reference",
)
```

### DO: Match INE points spatially
```python
# 1) Point inside building footprint -> direct match
# 2) No inside point -> average of 2 nearest points only if second point is within threshold
# 3) If 2-nearest fallback fails -> median of INE points within a fixed 50 m radius
# Reference labels: "INE" for direct/inside matches, "INE Assumption" for both fallback steps
```

### DO: Emit zone-helper enrichment summary logs
```python
print(
    "Zone-helper height enrichment summary: "
    "INE=..., INE Assumption=..., CEA Assumption replaced=..., heights changed=..."
)
```

### DO: Keep `height_ag` and `floors_ag` valid
```python
# Guard required by CEA geometry checks
floors_ag = max(1, min(floors_ag, floor(height_ag)))
```

### DO: Assign per-building defaults as sequences
```python
# When a full-column default is needed, use one value per row.
shapefile["building:levels"] = [3] * no_buildings
```

### DON'T: Scale per-building defaults by row count
```python
# Wrong: this creates inflated levels (e.g., 42 floors for 14 buildings)
shapefile["building:levels"] = 3 * no_buildings
```

### DO: Apply strict surroundings guard after enrichment
```python
result = apply_surroundings_floor_validity_guard(result)
```

### DON'T: Use address strings for matching
```python
# Avoid matching by MORADA/OSM address text - use geometry only
```

## Related Files
- `zone_helper.py` - Zone geometry + attributes from OSM.
- `surroundings_helper.py` - Surroundings geometry + attributes from OSM.
- `height_enrichment.py` - Shared INE height enrichment logic.
- `../default.config` - Parameters:
  - `[zone-helper] building-height-gpkg`
  - `[zone-helper] ine-fallback-max-distance-m`
  - `[surroundings-helper] building-height-gpkg`
  - `[surroundings-helper] ine-fallback-max-distance-m`
