"""
Geometry generator from
Shapefiles (building footprint)
and .tiff (terrain)

into 3D geometry with windows and roof equivalent to LOD3

"""
from __future__ import annotations
import math
import os
import pickle
import time
from itertools import repeat
import geopandas as gpd
import numpy as np
from cea.datamanagement.databases_verification import verify_input_geometry_zone, verify_input_geometry_surroundings # I added this
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import triangulate as shapely_triangulate
import py4design.py3dmodel.calculate as calculate
import py4design.py3dmodel.construct as construct
import py4design.py3dmodel.fetch as fetch
import py4design.py3dmodel.modify as modify
import py4design.py3dmodel.utility as utility
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.gp import gp_Pnt, gp_Lin, gp_Ax1, gp_Dir
from osgeo import osr, gdal
from py4design import urbangeom
from typing import TYPE_CHECKING, List, Tuple, Literal

import cea
import cea.config
import cea.inputlocator
import cea.utilities.parallel

if TYPE_CHECKING:
    import geopandas as gpd
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid
    import shapely

__author__ = "Jimeno A. Fonseca"
__copyright__ = "Copyright 2017, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Jimeno A. Fonseca", "Kian Wee Chen"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

from cea.utilities.standardize_coordinates import (get_lat_lon_projected_shapefile, get_projected_coordinate_system,
                                                   crs_to_epsg)
from collections import defaultdict

SURFACE_TYPES = ['walls', 'windows', 'roofs', 'undersides']
SURFACE_DIRECTION_LABELS = {'windows_east',
                            'windows_west',
                            'windows_south',
                            'windows_north',
                            'walls_east',
                            'walls_west',
                            'walls_south',
                            'walls_north',
                            'roofs_top',
                            'undersides_bottom',
                            }


def identify_surfaces_type(occface_list: List[TopoDS_Face]) -> Tuple[List[TopoDS_Face], 
                                                                     List[TopoDS_Face], 
                                                                     List[TopoDS_Face], 
                                                                     List[TopoDS_Face], 
                                                                     List[TopoDS_Face], 
                                                                     List[TopoDS_Face]]:
    roof_list = []
    footprint_list = []
    facade_list_north = []
    facade_list_west = []
    facade_list_east = []
    facade_list_south = []
    vec_vertical = (0, 0, 1)
    vec_horizontal = (0, 1, 0)

    # distinguishing between facade, roof and footprint.
    for f in occface_list:
        # get the normal of each face
        n = calculate.face_normal(f)
        flatten_n = [n[0], n[1], 0]  # need to flatten to erase Z just to consider vertical surfaces.
        angle_to_vertical = calculate.angle_bw_2_vecs(vec_vertical, n)
        # means its a facade
        if angle_to_vertical > 45 and angle_to_vertical < 135:
            angle_to_horizontal = calculate.angle_bw_2_vecs_w_ref(vec_horizontal, flatten_n, vec_vertical)
            if (0 <= angle_to_horizontal <= 45) or (315 <= angle_to_horizontal <= 360):
                facade_list_north.append(f)
            elif (45 < angle_to_horizontal < 135):
                facade_list_west.append(f)
            elif (135 <= angle_to_horizontal <= 225):
                facade_list_south.append(f)
            elif 225 < angle_to_horizontal < 315:
                facade_list_east.append(f)
        elif angle_to_vertical <= 45:
            roof_list.append(f)
        elif angle_to_vertical >= 135:
            footprint_list.append(f)

    return facade_list_north, facade_list_west, facade_list_east, facade_list_south, roof_list, footprint_list


def calc_intersection(surface, edges_coords, edges_dir, tolerance):
    """
    This script calculates the intersection of the building edges to the terrain,
    """
    point = gp_Pnt(edges_coords[0], edges_coords[1], edges_coords[2])
    direction = gp_Dir(edges_dir[0], edges_dir[1], edges_dir[2])
    line = gp_Lin(gp_Ax1(point, direction))

    terrain_intersection_curves = IntCurvesFace_ShapeIntersector()
    terrain_intersection_curves.Load(surface, tolerance)
    terrain_intersection_curves.PerformNearest(line, float("-inf"), float("+inf"))

    if terrain_intersection_curves.IsDone():
        npts = terrain_intersection_curves.NbPnt()
        if npts != 0:
            return terrain_intersection_curves.Pnt(1), terrain_intersection_curves.Face(1)
        else:
            return None, None
    else:
        return None, None


def create_windows(surface: TopoDS_Face, 
                   wwr: float, 
                   ref_pypt: Tuple[float, float, float],
                   ) -> TopoDS_Face:
    """
    This function creates a window by schrinking the surface according to the wwr around the reference point.
    The generated window has the same shape as the original surface.
    Each side of the surface is shrunk by `sqrt(wwr)`, so that the area is scaled down to `wwr * A_surface`.

    :param surface: the surface to be shrunk.
    :type surface: OCCface
    :param wwr: Window-to-Wall Ratio: the ratio of the surface that will be a window, ranges from 0 to 1.
    :type wwr: float
    :param ref_pypt: the reference point for the scaling operation. Usually the center point of the surface.
    :type ref_pypt: tuple
    :return: the scaled surface which represents the window.
    :rtype: OCCface
    """
    scaler = math.sqrt(wwr)
    return fetch.topo2topotype(modify.uniform_scale(surface, scaler, scaler, scaler, ref_pypt))


def create_hollowed_facade(surface_facade: TopoDS_Face, 
                           window: TopoDS_Face,
                           ) -> Tuple[List[TopoDS_Face], TopoDS_Face]:
    """
    clips the raw surface_facade with the window to create a hollowed facade using boolean difference.
    This will generate two surfaces in a list, and the first one selected to be the hollowed facade.
    Then, the face will be transformed into a mesh (list of triangles). Small triangles will be removed from the list.

    :param surface_facade: the facade surface to be hollowed.
    :type surface_facade: OCCface
    :param window: the window surface to be subtracted from the facade, generated by `create_windows`.
    :type window: OCCface
    :return: the hollowed facade as a mesh (list of `OCCface` triangles).
    :rtype: list[OCCface]
    :return: the hollowed facade created by the window subtraction.
    :rtype: OCCface
    """
    b_facade_cmpd = fetch.topo2topotype(construct.boolean_difference(surface_facade, window))
    hole_facade = fetch.topo_explorer(b_facade_cmpd, "face")[0]
    hollowed_facade = construct.simple_mesh(hole_facade)
    # Clean small triangles: this is a despicable source of error
    hollowed_facade_clean = [x for x in hollowed_facade if calculate.face_area(x) > 1E-3]

    return hollowed_facade_clean, hole_facade


def calc_building_solids(buildings_df: gpd.GeoDataFrame, 
                         geometry_simplification: float, 
                         elevation_map: ElevationMap, 
                         num_processes: int,
                         ) -> Tuple[List[TopoDS_Solid], List[float]]:
    """create building solids respecting their elevation, from building GeoDataFrame and elevation map.

    :param buildings_df: either the zone or surroundings buildings dataframe. 
        It should be read from either `"\\scenario\\inputs\\building-geometry\\zone.shp"` 
        or `"\\scenario\\inputs\\building-geometry\\surroundings.shp"`, 
        and the index of this GeoDataFrame should be the building names.
    :type buildings_df: GeoDataFrame
    :param geometry_simplification: the tolerance of simplification. 
        "All parts of a simplified geometry will be no more than tolerance distance from the original." 
        (from `geopandas.GeoSeries.simplify`)
    :type geometry_simplification: float
    :param elevation_map: an instance of `ElevationMap` that contains the terrain elevation data, read from a raster file 
    `"\\scenario\\inputs\\topography\\terrain.tif"`.
    :type elevation_map: ElevationMap
    :param num_processes: number of processes to use for parallel computation.
    :type num_processes: int
    :return: a list of solids representing the building external shells, each made from footprint + vertical external walls + roof.
    :rtype: list[OCCsolid]
    :return: a list of elevations corresponding to each building solid, where each elevation is a float value.
    :rtype: list[float]
    """

    # simplify geometry for buildings of interest
    geometries = buildings_df.geometry.simplify(geometry_simplification, preserve_topology=True)

    height = buildings_df['height_ag'].astype(float)
    nfloors = buildings_df['floors_ag'].astype(int)
    void_decks = buildings_df['void_deck'].astype(int)
    if not all(void_decks <= nfloors):
        raise ValueError(f"Void deck values must be less than or equal to the number of floors for each building. "
                         f"Found void_deck values: {void_decks.values} and number of floors: {nfloors.values}.")
    
    range_floors = [range(void_deck, floors + 1) for void_deck, floors in zip(void_decks, nfloors)]
    floor_to_floor_height = height / nfloors

    n = len(geometries)
    out = cea.utilities.parallel.vectorize(process_geometries, num_processes,
                                           on_complete=print_terrain_intersection_progress)(
        geometries, repeat(elevation_map, n), range_floors, floor_to_floor_height)

    solids, elevations = zip(*out)
    return list(solids), list(elevations)

def process_geometries(geometry: shapely.Polygon, 
                       elevation_map: ElevationMap, 
                       range_floors: range, 
                       floor_to_floor_height: float,
                       ) -> Tuple[TopoDS_Solid, float]:
    """
    gets the 2D geometry as well as the height and number of floors, and returns a solid representing the building. 
    Also returns the elevation of the building footprint using elevation_map.

    :param geometry: one building geometry from the buildings GeoDataFrame.
    :type geometry: shapely.Polygon
    :param elevation_map: the elevation map for the whole site.
    :type elevation_map: ElevationMap
    :param range_floors: range of floors for the building. For example, a building of 3 floors will have `range(4) = [0, 1, 2, 3]`.
    :type range_floors: range
    :param floor_to_floor_height: the height of each level of the building. 
    :type floor_to_floor_height: float
    :return: a solid representing the building, made from footprint + vertical external walls + roof.
    :rtype: OCCsolid
    :return: the elevation of the terrain at the footprint of the building.
    :rtype: float
    """
    elevation_map_for_geometry = elevation_map.get_elevation_map_from_geometry(geometry)
    # burn buildings footprint into the terrain and return the location of the new face
    face_footprint, elevation = burn_buildings(geometry, elevation_map_for_geometry, 1e-12)
    # create floors and form a solid
    building_solid = calc_solid(face_footprint, range_floors, floor_to_floor_height)

    return building_solid, elevation


def calc_building_geometry_surroundings(name: str, 
                                        building_solid: TopoDS_Solid, 
                                        geometry_pickle_dir: str,
                                        ) -> str:
    facade_list, roof_list, footprint_list = urbangeom.identify_building_surfaces(building_solid)
    geometry_3D_surroundings = {"name": name,
                                "windows": [],
                                "walls": facade_list,
                                "roofs": roof_list,
                                "footprint": footprint_list,
                                "orientation_walls": [],
                                "orientation_windows": [],
                                "normals_windows": [],
                                "normals_walls": [],
                                "intersect_walls": []}

    building_geometry = BuildingGeometry(**geometry_3D_surroundings)
    building_geometry.save(os.path.join(geometry_pickle_dir, 'surroundings', str(name)))
    return name

def _iter_polygons(geometry):
    """Yield Polygon objects from Polygon / MultiPolygon geometry."""
    if geometry is None:
        return
    geom_type = getattr(geometry, "geom_type", None)
    if geom_type == "Polygon":
        yield geometry
    elif geom_type == "MultiPolygon":
        for poly in geometry.geoms:
            yield poly


def _project_points_to_best_fit_plane(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]] | None:
    """
    Project polygon vertices to a single best-fit plane to guarantee planarity for OCC face creation.
    """
    if len(points) < 3:
        return None

    points_array = np.asarray(points, dtype=float)
    if points_array.shape[1] != 3:
        return None

    centroid = points_array.mean(axis=0)
    centered = points_array - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 3:
        return None

    normal = vh[-1]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        return None
    normal = normal / normal_norm

    distances = centered @ normal
    projected = points_array - np.outer(distances, normal)
    return [tuple(float(v) for v in point) for point in projected]


def _best_fit_plane_frame(points: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return a best-fit plane frame (centroid, axis_u, axis_v, normal) for 3D points.
    """
    if len(points) < 3:
        return None
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        return None
    centroid = points_array.mean(axis=0)
    centered = points_array - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 3:
        return None

    axis_u = vh[0]
    normal = vh[-1]
    axis_v = np.cross(normal, axis_u)

    norm_u = float(np.linalg.norm(axis_u))
    norm_v = float(np.linalg.norm(axis_v))
    norm_n = float(np.linalg.norm(normal))
    if norm_u <= 1e-12 or norm_v <= 1e-12 or norm_n <= 1e-12:
        return None
    axis_u = axis_u / norm_u
    axis_v = axis_v / norm_v
    normal = normal / norm_n
    return centroid, axis_u, axis_v, normal


def _remove_duplicate_vertices(points: list[tuple[float, float, float]], tol: float = 1e-7) -> list[tuple[float, float, float]]:
    """
    Remove consecutive duplicate points (and repeated closure point) from an xyz ring.
    """
    if not points:
        return []
    deduped: list[tuple[float, float, float]] = []
    for point in points:
        if not deduped:
            deduped.append(point)
            continue
        prev = deduped[-1]
        if (
            abs(point[0] - prev[0]) <= tol
            and abs(point[1] - prev[1]) <= tol
            and abs(point[2] - prev[2]) <= tol
        ):
            continue
        deduped.append(point)
    if len(deduped) >= 2:
        first, last = deduped[0], deduped[-1]
        if (
            abs(first[0] - last[0]) <= tol
            and abs(first[1] - last[1]) <= tol
            and abs(first[2] - last[2]) <= tol
        ):
            deduped = deduped[:-1]
    return deduped


def _build_occ_face_from_points(points_xyz: list[tuple[float, float, float]]):
    """
    Build one OCC face from ordered xyz ring vertices (without closure point).
    Returns None if OCC rejects geometry.
    """
    if len(points_xyz) < 3:
        return None
    closed = list(points_xyz) + [points_xyz[0]]
    try:
        face = construct.make_polygon(closed)
    except Exception:
        return None

    try:
        n = calculate.face_normal(face)
    except Exception:
        return None
    if n[2] < 0:
        closed = list(reversed(closed))
        try:
            face = construct.make_polygon(closed)
        except Exception:
            return None

    try:
        area = float(calculate.face_area(face))
    except Exception:
        return None
    if not math.isfinite(area) or area <= 1e-9:
        return None
    return face


def _triangulate_complex_roof(points_xyz: list[tuple[float, float, float]]) -> list:
    """
    Fallback for complex / OCC-failing roof polygons:
    triangulate in a local 2D best-fit plane and lift triangles back to 3D.
    """
    frame = _best_fit_plane_frame(points_xyz)
    if frame is None:
        return []
    centroid, axis_u, axis_v, _ = frame

    def to_uv(point_xyz):
        p = np.asarray(point_xyz, dtype=float) - centroid
        return float(np.dot(p, axis_u)), float(np.dot(p, axis_v))

    uv_points = [to_uv(point) for point in points_xyz]
    polygon_2d = ShapelyPolygon(uv_points)
    if polygon_2d.is_empty or polygon_2d.area <= 1e-9:
        return []
    if not polygon_2d.is_valid:
        polygon_2d = polygon_2d.buffer(0)
    if polygon_2d.is_empty or polygon_2d.area <= 1e-9:
        return []

    faces = []
    for tri in shapely_triangulate(polygon_2d):
        if tri.is_empty or tri.area <= 1e-9:
            continue
        # Keep only triangles truly inside the source polygon.
        intersection_area = tri.intersection(polygon_2d).area
        if intersection_area <= 1e-9:
            continue
        if intersection_area / tri.area < 0.99:
            continue

        tri_uv = list(tri.exterior.coords)
        if len(tri_uv) < 4:
            continue
        tri_uv = tri_uv[:-1]  # remove closure
        tri_xyz = []
        for u, v in tri_uv:
            xyz = centroid + axis_u * float(u) + axis_v * float(v)
            tri_xyz.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

        face = _build_occ_face_from_points(tri_xyz)
        if face is not None:
            faces.append(face)

    return faces


def _polygon_to_occ_faces(poly):
    """
    Convert a shapely Polygon with 3D coords to one or more OCC faces.
    Returns (faces, mode) where mode is one of: direct, triangulated, skipped.
    """
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return [], "skipped"

    # Require XYZ input for roof tilt/orientation
    if len(coords[0]) < 3:
        return [], "skipped"

    raw_points = [(float(x), float(y), float(z)) for x, y, z, *_ in coords]

    # Remove ring closure point before geometric processing.
    raw_points = _remove_duplicate_vertices(raw_points)
    if len(raw_points) < 3:
        return [], "skipped"

    planar_points = _project_points_to_best_fit_plane(raw_points)
    if planar_points is None or len(planar_points) < 3:
        return [], "skipped"

    face = _build_occ_face_from_points(planar_points)
    if face is not None:
        return [face], "direct"

    triangles = _triangulate_complex_roof(planar_points)
    if triangles:
        return triangles, "triangulated"

    return [], "skipped"

def load_custom_roof_faces(config, target_crs):
    """
    Load optional roof surfaces from:
    <scenario>/inputs/building-geometry/roof_surfaces.geojson

    Expected columns:
    - building (str): building name matching zone 'name'
    - geometry: Polygon/MultiPolygon with XYZ coords
    """

    roof_file = os.path.join(
        config.scenario, "inputs", "building-geometry", "roof_surfaces.geojson"
    )

    if not os.path.exists(roof_file):
        print(f"No custom roof file found at {roof_file}. Using default flat roofs.")
        return {}

    roof_gdf = gpd.read_file(roof_file)
    if roof_gdf.empty:
        print("Custom roof file is empty. Using default flat roofs.")
        return {}

    if "building" not in roof_gdf.columns:
        raise ValueError("roof_surfaces.geojson must contain a 'building' column.")

    if roof_gdf.crs is not None and target_crs is not None and roof_gdf.crs != target_crs:
        roof_gdf = roof_gdf.to_crs(target_crs)

    roofs_by_building = defaultdict(list)
    skipped = 0
    loaded_direct = 0
    loaded_triangulated = 0
    triangulated_source_polygons = 0

    for _, row in roof_gdf.iterrows():
        bname = row["building"]
        if pd.isna(bname):
            skipped += 1
            continue

        for poly in _iter_polygons(row.geometry):
            faces, mode = _polygon_to_occ_faces(poly)
            if not faces:
                skipped += 1
                continue
            if mode == "triangulated":
                triangulated_source_polygons += 1
                loaded_triangulated += len(faces)
            else:
                loaded_direct += len(faces)
            roofs_by_building[str(bname)].extend(faces)

    print(
        "Loaded custom roofs for "
        f"{len(roofs_by_building)} buildings. "
        f"Direct faces: {loaded_direct}. "
        f"Triangulated faces: {loaded_triangulated} "
        f"(from {triangulated_source_polygons} source polygons). "
        f"Skipped source polygons: {skipped}."
    )
    return dict(roofs_by_building)


def _face_z_coordinates(face: TopoDS_Face) -> list[float]:
    points = fetch.points_frm_occface(face)
    z_values = []
    for point in points:
        if len(point) >= 3:
            z_values.append(float(point[2]))
    return z_values


def _min_roof_z(roof_faces: list[TopoDS_Face]) -> float | None:
    if not roof_faces:
        return None
    z_values = []
    for face in roof_faces:
        z_values.extend(_face_z_coordinates(face))
    if not z_values:
        return None
    return float(min(z_values))


def _translate_faces_in_z(roof_faces: list[TopoDS_Face], delta_z: float) -> list[TopoDS_Face]:
    if abs(delta_z) <= 1e-9:
        return list(roof_faces)

    source = (0.0, 0.0, 0.0)
    destination = (0.0, 0.0, float(delta_z))
    translated = []
    for face in roof_faces:
        moved = modify.move(source, destination, face)
        translated.append(fetch.topo2topotype(moved))
    return translated


def align_custom_roofs_to_default_roof_level(
        custom_roofs: list[TopoDS_Face],
        default_roofs: list[TopoDS_Face],
) -> tuple[list[TopoDS_Face], float]:
    """
    Align custom roof faces so their base rests on the default OSM-derived roof level.

    The default roof level comes from the original extruded building solid (before custom roof replacement).
    """
    default_roof_min_z = _min_roof_z(default_roofs)
    custom_roof_min_z = _min_roof_z(custom_roofs)

    if default_roof_min_z is None or custom_roof_min_z is None:
        return list(custom_roofs), 0.0

    delta_z = float(default_roof_min_z - custom_roof_min_z)
    return _translate_faces_in_z(custom_roofs, delta_z), delta_z



def building_2d_to_3d(zone_df, surroundings_df, architecture_wwr_df, elevation_map, config,
                      geometry_pickle_dir, custom_roofs_by_building):

    """reconstruct 3D building geometries with windows and store each building's 3D data into a file.

    :param zone_df: data and 2D geometry of all analyzed building in the site, typically read from `zone.shp`.
    :type zone_df: GeoDataFrame
    :param surroundings_df: data and 2D geometry of surrounding buildings.
    :type surroundings_df: GeoDataFrame
    :param architecture_wwr_df: building envelope data.
    :type architecture_wwr_df: DataFrame
    :param elevation_map: 3D elevation map of the site.
    :type elevation_map: ElevationMap
    :param config: Configuration object with the settings (general and radiation)
    :type config: cea.config.Configuration
    :param geometry_pickle_dir: directory for saving building's 3D data.
    :type geometry_pickle_dir: str
    :return: names of analyzed buildings.
    :rtype: list[str]
    :return: names of surrounding buildings.
    :rtype: list[str]

    .. note::

        the function will not return any useful data, but it stores the reconstructed 
        building geometry (including envelopes, windows, their orientation 
        and if they are intersected with other solids) into a file that could be used in other steps.
    """
    # Config variables
    num_processes = config.get_number_of_processes()
    zone_simplification = config.radiation.zone_geometry
    surroundings_simplification = config.radiation.surrounding_geometry
    neglect_adjacent_buildings = config.radiation.neglect_adjacent_buildings

    print('Calculating terrain intersection of building geometries')
    zone_buildings_df: pd.DataFrame = zone_df.set_index('name')
    print(zone_df.head(),'zone_df')
    print(zone_buildings_df.head(),'zone_buildings_df')

    # merge architecture wwr data into zone buildings dataframe with "name" column,
    # because we want to use void_deck when creating the building solid.
    zone_building_names = zone_buildings_df.index.values
    zone_building_solid_list, zone_elevations = calc_building_solids(zone_buildings_df, zone_simplification,
                                                                     elevation_map, num_processes)

    # Check if there are any buildings in surroundings_df before processing
    if not surroundings_df.empty:
        surroundings_buildings_df = surroundings_df.set_index('name')
        if 'void_deck' not in surroundings_buildings_df.columns:
            surroundings_buildings_df['void_deck'] = 0
            
        surroundings_building_names = surroundings_buildings_df.index.values
        surroundings_building_solid_list, _ = calc_building_solids(
            surroundings_buildings_df, surroundings_simplification, elevation_map, num_processes)
        # calculate geometry for the surroundings
        print('Generating geometry for surrounding buildings')
        geometry_3D_surroundings = [calc_building_geometry_surroundings(x, y, geometry_pickle_dir) for x, y in
                                    zip(surroundings_building_names, surroundings_building_solid_list)]
    else:
        surroundings_building_solid_list = []
        geometry_3D_surroundings = []

    # calculate geometry for the zone of analysis
    print('Generating geometry for buildings in the zone of analysis')
    n = len(zone_building_names)
    calc_zone_geometry_multiprocessing = cea.utilities.parallel.vectorize(calc_building_geometry_zone,
                                                                          num_processes,
                                                                          on_complete=print_progress)

    if not neglect_adjacent_buildings:
        all_building_solid_list = np.append(zone_building_solid_list, surroundings_building_solid_list)
    else:
        all_building_solid_list = []
    # TODO: maybe move calc_building_solid into this function and avoid using archiecture_wwr_df, because it's already merged into zone_buildings_df.
    geometry_3D_zone = calc_zone_geometry_multiprocessing(zone_building_names,
                                                      zone_building_solid_list,
                                                      repeat(all_building_solid_list, n),
                                                      repeat(architecture_wwr_df, n),
                                                      repeat(custom_roofs_by_building, n),
                                                      repeat(geometry_pickle_dir, n),
                                                      repeat(neglect_adjacent_buildings, n),
                                                      zone_elevations)



    return geometry_3D_zone, geometry_3D_surroundings


def print_progress(i, n, _, __):
    current = i + 1
    if current == 1 or current == n or current % 100 == 0:
        print("Generating geometry for building {i} completed out of {n}".format(i=current, n=n))


def print_terrain_intersection_progress(i, n, _, __):
    current = i + 1
    if current == 1 or current == n or current % 100 == 0:
        print("Creating geometry for building {i} completed out of {n}".format(i=current, n=n))


def are_buildings_close_to_eachother(x_1, y_1, solid2, dist=100):
    box2 = calculate.get_bounding_box(solid2)
    x_2 = box2[0]
    y_2 = box2[1]
    delta = math.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2)
    if delta <= dist:
        return True
    else:
        return False


class BuildingGeometry(object):
    __slots__ = ["name", "terrain_elevation", "footprint",
                 "windows",    "orientation_windows",    "normals_windows",    "intersect_windows",
                 "walls",      "orientation_walls",      "normals_walls",      "intersect_walls",
                 "roofs",      "orientation_roofs",      "normals_roofs",      "intersect_roofs",
                 "undersides", "orientation_undersides", "normals_undersides", "intersect_undersides",
                 ]
    # footprint means the building's projection on the ground, 
    # undersides means the bottom surface of the building.

    def __init__(self, **kwargs):
        for key in self.__slots__:
            value = kwargs.get(key)
            setattr(self, key, value)

    def __getstate__(self):
        return [getattr(self, k, None) for k in self.__slots__]

    def __setstate__(self, data):
        for k, v in zip(self.__slots__, data):
            setattr(self, k, v)

    @classmethod
    def load(cls, pickle_location):
        with open(pickle_location, 'rb') as fp:
            values = pickle.load(fp)
            obj = cls()
            obj.__setstate__(values)
        return obj

    def save(self, pickle_location):
        dir_name = os.path.dirname(pickle_location)
        os.makedirs(dir_name, exist_ok=True)

        with open(pickle_location, 'wb') as f:
            pickle.dump(self.__getstate__(), f)
        return pickle_location


def calc_building_geometry_zone(name,
                                building_solid,
                                all_building_solid_list,
                                architecture_wwr_df,
                                custom_roofs_by_building,
                                geometry_pickle_dir,
                                neglect_adjacent_buildings,
                                elevation):

    """_summary_

    :param name: name of building.
    :type name: str
    :param building_solid: 
        a closed geometry representing the building external shells, 
        made from footprint + vertical external walls + roof (without windows).
    :type building_solid: OCCsolid
    :param all_building_solid_list: a list of all_building_solid_list, or an empty list if `neglect_adjacent_buildings == True`.
    :type all_building_solid_list: list[OCCsolid]
    :param architecture_wwr_df: a dataframe read from `locator.get_building_architecture` containing envelope info.
    :type architecture_wwr_df: DataFrame
    :param geometry_pickle_dir: folder path to save the created `BuildingGeometry` object.
    :type geometry_pickle_dir: str
    :param neglect_adjacent_buildings: True if no adjacency of other buildings is considered.
    :type neglect_adjacent_buildings: bool
    :param elevation: elevation of building footprint's middle point.
    :type elevation: float
    :return: name of building.
    :rtype: str

    .. note::

        the function will not return any useful data, but it stores the reconstructed 
        building geometry (including envelopes, windows, their orientation 
        and if they are intersected with other solids) into a file that could be used in other steps.
    """
    # now get all surfaces and create windows only if the buildings are in the area of study
    window_list = []
    wall_list = []
    orientation = []
    orientation_win = []
    normals_walls = []
    normals_win = []
    intersect_wall = []

    # check if buildings are close together and it merits to check the intersection
    # close together is defined as:
    # if two building bounding boxes' southwest corner are smaller than 100m on their xy-plane projection (ignoring z-axis).
    # TODO: maybe also check if one building's top roof is within another building's volume when two buildings are stacked.
    potentially_intersecting_solids = []
    if not neglect_adjacent_buildings:
        box = calculate.get_bounding_box(building_solid)
        x, y = box[0], box[1]
        for solid in all_building_solid_list:
            if are_buildings_close_to_eachother(x, y, solid):
                potentially_intersecting_solids.append(solid)

    # identify building surfaces according to angle:
    face_list = fetch.faces_frm_solid(building_solid)
    facade_list_north, facade_list_west, \
    facade_list_east, facade_list_south, roof_list, footprint_list = identify_surfaces_type(face_list)
    default_roof_list = list(roof_list)

    custom_roofs = custom_roofs_by_building.get(name, [])
    if custom_roofs:
        roof_list, z_shift_m = align_custom_roofs_to_default_roof_level(custom_roofs, default_roof_list)
        if abs(z_shift_m) > 1e-6:
            print(f"Aligned custom roofs for building {name} by {z_shift_m:.3f} m to match OSM roof level.")
        print(f"Using {len(roof_list)} custom roof faces for building {name}")



    # get window properties
    wwr_west = float(architecture_wwr_df.loc[name, "wwr_west"])
    wwr_east = float(architecture_wwr_df.loc[name, "wwr_east"])
    wwr_north = float(architecture_wwr_df.loc[name, "wwr_north"])
    wwr_south = float(architecture_wwr_df.loc[name, "wwr_south"])

    def process_facade(facade_list, wwr, orientation_label):
        window, wall, normals_window, normals_wall, wall_intersects = calc_windows_walls(
            facade_list, wwr, potentially_intersecting_solids)
        if len(window) != 0:
            window_list.extend(window)
            orientation_win.extend([orientation_label] * len(window))
            normals_win.extend(normals_window)
        wall_list.extend(wall)
        orientation.extend([orientation_label] * len(wall))
        normals_walls.extend(normals_wall)
        intersect_wall.extend(wall_intersects)

    process_facade(facade_list_west, wwr_west, 'west')
    process_facade(facade_list_east, wwr_east, 'east')
    process_facade(facade_list_north, wwr_north, 'north')
    process_facade(facade_list_south, wwr_south, 'south')

    intersect_windows = [0] * len(window_list)

    _, _, _, normals_roof, intersect_roof = calc_windows_walls(roof_list, 0.0, potentially_intersecting_solids)
    orientation_roofs = ["top"] * len(roof_list)

    _, _, _, normals_footprint, intersect_footprint = calc_windows_walls(footprint_list, 0.0, potentially_intersecting_solids)
    orientation_footprint = ["bottom"] * len(footprint_list)

    geometry_3D_zone = {"name": name, "footprint": footprint_list,
                        "windows": window_list,       "orientation_windows": orientation_win,          "normals_windows": normals_win,          "intersect_windows": intersect_windows,
                        "walls": wall_list,           "orientation_walls": orientation,                "normals_walls": normals_walls,          "intersect_walls": intersect_wall,
                        "roofs": roof_list,           "orientation_roofs": orientation_roofs,          "normals_roofs": normals_roof,           "intersect_roofs": intersect_roof,
                        "undersides": footprint_list, "orientation_undersides": orientation_footprint, "normals_undersides": normals_footprint, "intersect_undersides": intersect_footprint, 
                        }
    # see class BuildingGeometry for the difference between footprint and undersides.

    building_geometry = BuildingGeometry(**geometry_3D_zone, terrain_elevation=elevation)
    building_geometry.save(os.path.join(geometry_pickle_dir, 'zone', str(name)))
    return name


def burn_buildings(geometry: shapely.Polygon, 
                   elevation_map: ElevationMap, 
                   tolerance: float,
                   ) -> Tuple[TopoDS_Face, float]:
    """find the elevation of building footprint polygon by intersecting the center point of the polygon 
    with the terrain elevation map, then move the polygon to that elevation.

    :param geometry: a polygon representing the building footprint.
    :type geometry: shapely.Polygon
    :param elevation_map: elevation map that contains the building footprint polygon.
    :type elevation_map: ElevationMap
    :param tolerance: The minimal surface area of each triangulated face. Any faces smaller than the tolerance will be deleted.
    :type tolerance: float
    :return: the bottom surface of the building.
    :rtype: OCCface
    :return: the elevation of the terrain at the footprint of the building.
    :rtype: float
    """
    if geometry.has_z:
        # remove elevation - we'll add it back later by intersecting with the topography
        point_list_2D = ((a, b) for (a, b, _) in geometry.exterior.coords)
    else:
        point_list_2D = geometry.exterior.coords
    point_list_3D = [(a, b, 0) for (a, b) in point_list_2D]  # add 0 elevation

    # creating floor surface in pythonocc
    face = construct.make_polygon(point_list_3D)
    # get the midpt of the face
    face_midpt = calculate.face_midpt(face)

    terrain_tin = elevation_map.generate_tin(tolerance)
    # make shell out of tin_occface_list and create OCC object
    terrain_shell = construct.make_shell(terrain_tin)

    # project the face_midpt to the terrain and get the elevation
    inter_pt, inter_face = calc_intersection(terrain_shell, face_midpt, (0, 0, 1), tolerance)

    # reconstruct the footprint with the elevation
    loc_pt = (inter_pt.X(), inter_pt.Y(), inter_pt.Z())
    face = fetch.topo2topotype(modify.move(face_midpt, loc_pt, face))
    return face, inter_pt.Z()


def calc_solid(face_footprint: TopoDS_Face, 
               range_floors: range, 
               floor_to_floor_height: float,
               ) -> TopoDS_Solid:
    """
    extrudes the footprint surface into a 3D solid.

    :param face_footprint: footprint of the building. 
    :type face_footprint: OCCface
    :param range_floors: 
        range of floors for the building. 
        For example, a building of 3 floors will have `range(4) = [0, 1, 2, 3]`, 
        because it has 4 floors (1 ground + 2 middle + 1 roof).
    :type range_floors: range
    :param floor_to_floor_height: the height of each level of the building.
        For example, if the building has 3 floors and a height of 9m, then `floor_to_floor_height = 9 / 3 = 3`.
    :type floor_to_floor_height: float
    :return: a solid representing the building, made from footprint + vertical external walls + roof.
    :rtype: OCCsolid
    """
    # create faces for every floor and extrude the solid

    def cal_face_list(floor_counter):
        """
        This function moves the face_footprint to the correct height for a given floor_counter level.

        :param floor_counter: number of the floor to be offset. 0 stands for the ground floor.
        :type floor_counter: int
        :return: vertically offset face on the given floor height (0m for ground floor)
        :rtype: OCCface
        """
        dist2mve = floor_counter * floor_to_floor_height
        # get midpt of face
        orig_pt = calculate.face_midpt(face_footprint)
        # move the pt 1 level up
        dest_pt = modify.move_pt(orig_pt, (0, 0, 1), dist2mve)
        moved_face = modify.move(orig_pt, dest_pt, face_footprint)

        return moved_face

    moved_face_list = np.vectorize(cal_face_list)(range_floors)
    # make checks to satisfy a closed geometry also called a shell

    vertical_shell = construct.make_loft(moved_face_list)
    vertical_face_list = fetch.topo_explorer(vertical_shell, "face")
    roof = moved_face_list[-1]
    footprint = moved_face_list[0]
    all_faces = []
    all_faces.append(footprint)
    all_faces.extend(vertical_face_list)
    all_faces.append(roof)
    building_shell_list = construct.sew_faces(all_faces)

    # make sure all the normals are correct (they are pointing out)
    bldg_solid = construct.make_solid(building_shell_list[0])
    bldg_solid = modify.fix_close_solid(bldg_solid)
    return bldg_solid


class Points(object):
    def __init__(self, point_to_evaluate):
        self.point_to_evaluate = point_to_evaluate


def calc_windows_walls(facade_list: List[TopoDS_Face], 
                       wwr: float, 
                       potentially_intersecting_solids: List[TopoDS_Solid],
                       ) -> Tuple[List[TopoDS_Face], 
                                  List[TopoDS_Face], 
                                  List[Tuple[float, float, float]], 
                                  List[Tuple[float, float, float]], 
                                  List[Literal[0, 1]]]:
    """
    Classify each façade face as window or wall, generate any required geometry 
    (triangulated wall panels, punched windows), and return normals plus an intersection flag.

    :param facade_list: a list of vertical faces representing the facades of a building.
    :type facade_list: list[OCCface]
    :param wwr: window to wall ratio, ranges from 0 to 1.
    :type wwr: float
    :param potentially_intersecting_solids: 
        all buildings from the site that might intersect with the currently analysed building facade. 
        It could be an empty list, but if it's not, it will contain the building itself as well.
    :type potentially_intersecting_solids: list[OCCsolid]
    :raises ValueError: if wwr is not `float` or cannot be turned into `float`, raise ValueError.
    :return: windows on the facade surfaces.
    :rtype: list[OCCface]
    :return: opaque wall faces (triangulated where windows were created).
    :rtype: list[OCCface]
    :return: window surface normal vectors.
    :rtype: list[tuple[float, float, float]]
    :return: unit normals for every element in the previous *wall_list* 
    (duplicates exist because walls may be triangulated).
    :rtype: list[tuple[float, float, float]]
    :return: `1` if the original surface intersects with any other solid, or `0` otherwise.
    :rtype: list[Literal[0, 1]]
    """
    window_list = []
    wall_list = []
    normals_win = []
    normals_wall = []
    wall_intersects = []
    number_intersecting_solids = len(potentially_intersecting_solids)
    for surface_facade in facade_list:
        # get coordinates of surface
        ref_pypt = calculate.face_midpt(surface_facade)
        standard_normal = calculate.face_normal(surface_facade)  # to avoid problems with fuzzy normals

        # evaluate if the surface intersects any other solid (important to erase non-active surfaces in the building
        # simulation model)
        data_point = Points(modify.move_pt(ref_pypt, standard_normal, 0.1))

        if number_intersecting_solids:
            # flag weather it intersects a surrounding geometry
            intersects = np.vectorize(calc_intersection_face_solid)(potentially_intersecting_solids, data_point)
            intersects = sum(intersects)
        else:
            intersects = 0

        if intersects > 0:  # the face intersects so it is a partition wall, and no window is created
            wall_list.append(surface_facade)
            normals_wall.append(standard_normal)
            wall_intersects.append(1)
        else:
            try:    # Ensure wwr is a float
                wwr = float(wwr)
            except ValueError:
                raise ValueError(f"Invalid value for wwr: {wwr}. It must be a numeric value.")

            # offset the facade to create a window according to the wwr
            if 0.0 < wwr < 1.0:
                # for window
                window = create_windows(surface_facade, wwr, ref_pypt)
                window_list.append(window)
                normals_win.append(standard_normal)
                # for walls
                hollowed_facades, _ = create_hollowed_facade(surface_facade, window)  # accounts for hole created by window
                wall_intersects.extend([intersects] * len(hollowed_facades))
                wall_list.extend(hollowed_facades)
                normals_wall.extend([standard_normal] * len(hollowed_facades))

            elif wwr == 1.0:
                window_list.append(surface_facade)
                normals_win.append(standard_normal)
            else:
                wall_list.append(surface_facade)
                normals_wall.append(standard_normal)
                wall_intersects.append(intersects)

    return window_list, wall_list, normals_win, normals_wall, wall_intersects


def calc_intersection_face_solid(potentially_intersecting_solid, point):
    with cea.utilities.devnull():
        point_in_solid = calculate.point_in_solid(point.point_to_evaluate, potentially_intersecting_solid)
    if point_in_solid:
        intersects = 1
    else:
        intersects = 0
    return intersects


class ElevationMap(object):
    __slots__ = ['elevation_map', 'x_coords', 'y_coords', 'x_size', 'y_size', 'nodata']

    def __init__(self, elevation_map, x_coords, y_coords, x_size, y_size, nodata=None):
        self.elevation_map = elevation_map
        self.x_coords = x_coords
        self.y_coords = y_coords

        self.x_size = x_size
        self.y_size = y_size

        self.nodata = nodata

    @classmethod
    def read_raster(cls, raster):
        band = raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        a = band.ReadAsArray()

        y, x = np.shape(a)
        upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = raster.GetGeoTransform()

        if x_rotation != 0 or y_rotation != 0:
            raise ValueError("Rotation in raster is not supported.")

        x_coords = np.arange(start=0, stop=x) * x_size + upper_left_x + (x_size / 2)  # add half the cell size
        y_coords = np.arange(start=0, stop=y) * y_size + upper_left_y + (y_size / 2)  # to centre the point

        return cls(a, x_coords, y_coords, x_size, y_size, nodata)

    def get_elevation_map_from_geometry(self, geometry, extra_points=3):
        minx, miny, maxx, maxy = geometry.bounds

        # Ensure geometry bounds is within elevation map
        if (minx < self.x_coords[0] - self.x_size or maxx > self.x_coords[-1] + self.x_size
                or miny < self.y_coords[-1] + self.y_size or maxy > self.y_coords[0] - self.y_size):
            raise ValueError("Geometry bounds not within the elevation map.")

        x_start = np.searchsorted(self.x_coords - self.x_size, minx, side='left') - 1
        x_end = np.searchsorted(self.x_coords + self.x_size, maxx, side='right')
        y_start = len(self.y_coords) - np.searchsorted((self.y_coords - self.y_size)[::-1], maxy, side='left') - 1
        y_end = len(self.y_coords) - np.searchsorted((self.y_coords + self.y_size)[::-1], miny, side='right')

        # Consider extra points
        x_start = max(x_start - extra_points, 0)
        x_end = min(x_end + extra_points, len(self.x_coords))
        y_start = max(y_start - extra_points, 0)
        y_end = min(y_end + extra_points, len(self.y_coords))

        new_elevation_map = self.elevation_map[y_start:y_end + 1, x_start:x_end + 1]
        new_x_coords = self.x_coords[x_start:x_end + 1]
        new_y_coords = self.y_coords[y_start:y_end + 1]

        return ElevationMap(new_elevation_map, new_x_coords, new_y_coords, self.x_size, self.y_size, self.nodata)

    def generate_tin(self, tolerance=1e-6):
        """generates a 3D mesh from the elevation raster map.

        :param tolerance: The minimal surface area of each triangulated face. 
            Any faces smaller than the tolerance will be deleted. Defaults to `1e-6`.
        :type tolerance: float, optional
        :return: a list of OCCface triangles representing the 3D mesh of the terrain.
        :rtype: list[OCCface]
        """
        # Ignore no data values from raster
        y_index, x_index = np.nonzero(self.elevation_map != self.nodata)
        _x_coords = self.x_coords[x_index]
        _y_coords = self.y_coords[y_index]

        raster_points = ((x, y, z) for x, y, z in zip(_x_coords, _y_coords, self.elevation_map[y_index, x_index]))

        tin_occface_list = construct.delaunay3d(raster_points, tolerance=tolerance)

        return tin_occface_list


def standardize_coordinate_systems(zone_df, surroundings_df, trees_df, terrain_raster):
    # Change all to projected cr (to meters)
    lat, lon = get_lat_lon_projected_shapefile(zone_df)
    crs = get_projected_coordinate_system(lat, lon)

    reprojected_zone_df = zone_df.to_crs(crs)
    reprojected_surroundings_df = surroundings_df.to_crs(crs)
    reprojected_trees_df = trees_df.to_crs(crs)

    reprojected_terrain = gdal.Warp(
        '',  # Empty string as the output file path means in-memory
        terrain_raster,
        format='VRT',  # Use VRT format for in-memory operation
        dstSRS=crs
    )

    print(f"Reprojected scene to `EPSG:{crs_to_epsg(crs)}`")
    return reprojected_zone_df, reprojected_surroundings_df, reprojected_trees_df, reprojected_terrain


def check_terrain_bounds(zone_df, surroundings_df, trees_df, terrain_raster):
    total_df = pd.concat([zone_df, surroundings_df, trees_df])

    # minx, miny, maxx, maxy
    geometry_bounds = total_df.total_bounds

    upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size = terrain_raster.GetGeoTransform()
    minx = upper_left_x
    maxy = upper_left_y
    maxx = minx + x_size * terrain_raster.RasterXSize
    miny = maxy + y_size * terrain_raster.RasterYSize

    if minx > geometry_bounds[0] or miny > geometry_bounds[1] or maxx < geometry_bounds[2] or maxy < geometry_bounds[3]:
        raise ValueError('Terrain provided does not cover all geometries. '
                         'Bounds of terrain must be larger than the total bounds of the scene.')


def tree_geometry_generator(tree_df, terrain_raster):
    terrian_projection = terrain_raster.GetProjection()
    proj4_str = osr.SpatialReference(wkt=terrian_projection).ExportToProj4()
    tree_df = tree_df.to_crs(proj4_str)

    elevation_map = ElevationMap.read_raster(terrain_raster)

    from multiprocessing.pool import Pool
    from multiprocessing import cpu_count

    with Pool(cpu_count() - 1) as pool:
        surfaces = [
            fetch.faces_frm_solid(solid) for (solid, _) in pool.starmap(
                process_geometries, (
                    (geom, elevation_map, (0, 1), z) for geom, z in zip(tree_df['geometry'], tree_df['height_tc'])
                )
            )
        ]

    return surfaces


def geometry_main(config: cea.config.Configuration, 
                  zone_df: gpd.GeoDataFrame, 
                  surroundings_df: gpd.GeoDataFrame, 
                  trees_df: gpd.GeoDataFrame, 
                  terrain_raster: gdal.Dataset, 
                  architecture_wwr_df: pd.DataFrame, 
                  geometry_pickle_dir: str,
                  ) -> Tuple[List[TopoDS_Face], 
                             List[str], 
                             List[str], 
                             List[List[TopoDS_Face]]]:
    """reads the input data of a scenario, generates and stores 3D data of each building, 
    and generate 3D geometry of the terrain.

    :param config: Configuration object with the settings (general and radiation)
    :type config: cea.config.Configuration
    :param zone_df: data and 2D geometry of all analyzed building in the site, typically read from `zone.shp`.
    :type zone_df: gpd.GeoDataFrame
    :param surroundings_df: data and 2D geometry of surrounding buildings.
    :type surroundings_df: gpd.GeoDataFrame
    :param trees_df: geometry and data of trees in the site (if any).
    :type trees_df: gpd.GeoDataFrame
    :param terrain_raster: the raw terrain grayscale graph containing elevation information.
    :type terrain_raster: gdal.Dataset
    :param architecture_wwr_df: detailed envelope information of each building.
    :type architecture_wwr_df: pd.DataFrame
    :param geometry_pickle_dir: directory where building 3D geometry data is stored.
    :type geometry_pickle_dir: str
    :return: a list of OCCface triangles representing the 3D mesh of the terrain.
    :rtype: list[OCCface]
    :return: names of analyzed buildings within the scenario.
    :rtype: list[str]
    :return: names of surrounding buildings within the scenario.
    :rtype: list[str]
    :return: list of tree geometries (if any)
    :rtype: list[list[OCCface]]
    """
    print("Standardizing coordinate systems")
    zone_df, surroundings_df, trees_df, terrain_raster = standardize_coordinate_systems(
    zone_df, surroundings_df, trees_df, terrain_raster)

    custom_roofs_by_building = load_custom_roof_faces(config, zone_df.crs)

    # clear in case there are repeated buildings from zone in surroundings file
    filter_surrounding_buildings = ~surroundings_df["name"].isin(zone_df["name"])
    surroundings_df = surroundings_df[filter_surrounding_buildings]

    check_terrain_bounds(zone_df, surroundings_df, trees_df, terrain_raster)

    # Create a triangulated irregular network of terrain from raster
    print("Reading terrain geometry")
    elevation_map = ElevationMap.read_raster(terrain_raster)
    terrain_tin = elevation_map.generate_tin()

    # transform buildings 2D to 3D and add windows
    print("Creating 3D building surfaces")
    os.makedirs(geometry_pickle_dir, exist_ok=True)
    geometry_3D_zone, geometry_3D_surroundings = building_2d_to_3d(zone_df,surroundings_df,architecture_wwr_df,elevation_map,config,geometry_pickle_dir,custom_roofs_by_building)


    tree_surfaces = []
    if len(trees_df.geometry) > 0:
        print("Creating tree surfaces")
        tree_surfaces = tree_geometry_generator(trees_df, terrain_raster)

    return terrain_tin, geometry_3D_zone, geometry_3D_surroundings, tree_surfaces


if __name__ == '__main__':
    print("popo")
    config = cea.config.Configuration()
    locator = cea.inputlocator.InputLocator(scenario=config.scenario)
    settings = config.radiation
    # config: cea.config.Configuration

    zone_path = locator.get_zone_geometry()
    surroundings_path = locator.get_surroundings_geometry()

    print(f"zone: {zone_path}")
    print(f"surroundings: {surroundings_path}")

    zone_df = gpd.GeoDataFrame.from_file(zone_path)
    surroundings_df = gpd.GeoDataFrame.from_file(surroundings_path)

    verify_input_geometry_zone(zone_df)
    verify_input_geometry_surroundings(surroundings_df)


    trees_df = gpd.GeoDataFrame(geometry=[], crs=zone_df.crs)
    terrain_raster = gdal.Open(locator.get_terrain())
    architecture_wwr_df = gpd.GeoDataFrame.from_file(locator.get_building_architecture()).set_index('name')
    geometry_pickle_dir = os.path.join(locator.get_solar_radiation_folder(), "radiance_geometry_pickle")
    # run routine City GML LOD 1


    time1 = time.time()

    (geometry_terrain,
         zone_building_names,
         surroundings_building_names,
         tree_surfaces) = geometry_main(config,zone_df,surroundings_df,trees_df,terrain_raster,architecture_wwr_df,geometry_pickle_dir)

    # geometry_main returns building names; load the serialized geometry objects for visualization.
    geometry_3D_zone = [BuildingGeometry.load(os.path.join(geometry_pickle_dir, 'zone', str(name)))
                        for name in zone_building_names]
    geometry_3D_surroundings = [BuildingGeometry.load(os.path.join(geometry_pickle_dir, 'surroundings', str(name)))
                                for name in surroundings_building_names]

    # to visualize the results
    geometry_buildings = []
    geometry_buildings_nonop = []
    walls_intercept = [wall for building_geometry in geometry_3D_zone for wall, inter in
                       zip((building_geometry.walls or []), (building_geometry.intersect_walls or [])) if inter > 0]
    windows = [window for building_geometry in geometry_3D_zone for window in (building_geometry.windows or [])]
    walls = [wall for building_geometry in geometry_3D_zone for wall in (building_geometry.walls or [])]
    roofs = [roof for building_geometry in geometry_3D_zone for roof in (building_geometry.roofs or [])]
    footprint = [face for building_geometry in geometry_3D_zone for face in (building_geometry.footprint or [])]
    walls_s = [wall for building_geometry in geometry_3D_surroundings for wall in (building_geometry.walls or [])]
    windows_s = [window for building_geometry in geometry_3D_surroundings for window in (building_geometry.windows or [])]
    roof_s = [roof for building_geometry in geometry_3D_surroundings for roof in (building_geometry.roofs or [])]

    geometry_buildings_nonop.extend(windows)
    geometry_buildings_nonop.extend(windows_s)
    geometry_buildings.extend(walls)
    geometry_buildings.extend(roofs)
    geometry_buildings.extend(footprint)
    geometry_buildings.extend(walls_s)
    geometry_buildings.extend(roof_s)
    normals_terrain = calculate.face_normal_as_edges(geometry_terrain, 5)
    utility.visualise([geometry_terrain, geometry_buildings, geometry_buildings_nonop, walls_intercept],
                      ["GREEN", "WHITE", "BLUE", "RED"], backend="pyside6")  # install Wxpython

    utility.visualise([walls_intercept],
                      ["RED"], backend="pyside6")

    utility.visualise([walls],
                      ["RED"], backend="pyside6")

    utility.visualise([windows],
                      ["RED"], backend="pyside6")
