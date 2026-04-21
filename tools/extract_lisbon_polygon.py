"""Extract a CEA/UCEA-compatible Lisbon municipality polygon.

The source ``municipios.geojson`` is stored in a projected CRS. UCEA expects
GeoJSON coordinates as WGS84 lon/lat, and its polygon parser consumes a single
outer ring, so this script exports the largest Lisbon polygon part in EPSG:4326.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform


DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "municipios.geojson"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "lisbonpolygon.geojson"
LISBON_MUNICIPALITY_CODE = "1106"


def _source_crs(payload: dict) -> str:
    crs_name = (
        payload.get("crs", {})
        .get("properties", {})
        .get("name", "urn:ogc:def:crs:EPSG::32729")
    )
    crs_text = str(crs_name)
    if crs_text.startswith("urn:ogc:def:crs:EPSG::"):
        return f"EPSG:{crs_text.rsplit('::', 1)[-1]}"
    return crs_text


def _find_lisbon_feature(payload: dict) -> dict:
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        if str(properties.get("CCA_2")) == LISBON_MUNICIPALITY_CODE:
            return feature
    raise ValueError(f"Could not find Lisbon municipality CCA_2={LISBON_MUNICIPALITY_CODE}.")


def _largest_polygon(geometry) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda polygon: polygon.area)
    raise ValueError(f"Expected Polygon or MultiPolygon, got {geometry.geom_type}.")


def extract_lisbon_polygon(source_path: Path, output_path: Path) -> dict:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_crs = _source_crs(payload)
    feature = _find_lisbon_feature(payload)
    projected_geometry = shape(feature["geometry"])
    polygon = _largest_polygon(projected_geometry)

    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon_lat_polygon = transform(transformer.transform, polygon)

    output = {
        "type": "FeatureCollection",
        "name": "lisbonpolygon",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Lisboa",
                    "municipality_code": LISBON_MUNICIPALITY_CODE,
                    "source": source_path.name,
                    "source_crs": source_crs,
                    "crs": "EPSG:4326",
                    "ucea_compatible": True,
                },
                "geometry": mapping(lon_lat_polygon),
            }
        ],
    }

    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = extract_lisbon_polygon(args.source, args.output)
    ring = output["features"][0]["geometry"]["coordinates"][0]
    print(f"Wrote {args.output}")
    print(f"Exported {len(ring)} lon/lat coordinates in EPSG:4326.")


if __name__ == "__main__":
    main()
