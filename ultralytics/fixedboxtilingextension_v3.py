import argparse
import hashlib
import io
import json
import math
import os
import glob
import cv2
import folium
import gradio as gr
import numpy as np
import requests
from PIL import Image
from owslib.wmts import WebMapTileService
from pyproj import Transformer
from ultralytics import YOLO

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "cache")

def _resolve_default_model_path():
    local_candidate = os.path.join(SCRIPT_DIR, "best.pt")
    if os.path.exists(local_candidate):
        return local_candidate

    checked = {os.path.normcase(os.path.normpath(local_candidate))}
    current = SCRIPT_DIR
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            break

        pattern = os.path.join(parent, "*", "ultralytics", "best.pt")
        for candidate in sorted(glob.glob(pattern)):
            norm = os.path.normcase(os.path.normpath(candidate))
            if norm in checked:
                continue
            checked.add(norm)
            if os.path.exists(candidate):
                return candidate

        current = parent

    return local_candidate

DEFAULT_MODEL_PATH = _resolve_default_model_path()
DEFAULT_ZONE_SHP_PATH = r"C:\Users\Andre\cea-scenarios\test-tilt\scenario\inputs\building-geometry\zone.shp"

ZONE_SHP_PATH = DEFAULT_ZONE_SHP_PATH

## Listing of logic in the project
'''
We download high resolution satellite images. 
From this images we retrieve predictions with our model.
Then we retrieve the OSM buildings from openstreetmap.
Then we match these buildings.
We recreate a rooftop based on the building outline and the model prediction 
We render it_make_topology_line_layer
'''

def draw_azimuths_on_satellite(
    satellite_image: np.ndarray,
    buildings,          # list[EstimatedBuilding]
    top_left_corner,    # (tl_x, tl_y) in EPSG:3763
    res: float,         # metres per pixel
) -> np.ndarray:
    """
    Returns a copy of `satellite_image` (RGB uint8) with, for every detected
    building, one arrow per active roof face pointing in the azimuth direction.
 
    Face colours match traveler_copy_2.py:
        top    â†’ red
        right  â†’ blue
        bottom â†’ green
        left   â†’ yellow
    """
    BASE_AZ = {
        "top":    math.pi / 2,
        "right":  0.0,
        "bottom": -math.pi / 2,
        "left":   math.pi,
    }
    FACE_COLOR_BGR = {
        "top":    (0,   0,   255),   # red
        "right":  (255, 0,   0  ),   # blue
        "bottom": (0,   255, 0  ),   # green
        "left":   (0,   255, 255),   # yellow
    }
 
    tl_x, tl_y = top_left_corner
    annotated = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)
 
    for b in buildings:
        xmin, ymin, xmax, ymax = b.box_coords_in_epsg_3763
 
        # bounding-box centre in pixel coordinates
        cx_px = int(((xmin + xmax) / 2 - tl_x) / res)
        cy_px = int((tl_y - (ymin + ymax) / 2) / res)
 
        # arrow length proportional to box size
        box_w_px = max(1, int((xmax - xmin) / res))
        box_h_px = max(1, int((ymax - ymin) / res))
        arrow_len = int(max(box_w_px, box_h_px) * 0.45)
        thickness = max(1, arrow_len // 12)
 
        # thin white bounding box for reference
        x1 = int((xmin - tl_x) / res)
        y1 = int((tl_y - ymax) / res)
        x2 = int((xmax - tl_x) / res)
        y2 = int((tl_y - ymin) / res)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)
 
        for plane in b.planes_in_physical_dimensions:
            face = plane.face
            ori  = plane.orientation   # already sigmoid-transformed, in [0, 1]
 
            azimuth_rad = (ori - 0.5) * (math.pi / 2) + BASE_AZ[face]
 
            # In image space: x â†’ East, y â†’ South (flipped vs. math convention)
            # so dy uses -sin to convert from math â†’ image coordinates
            dx = math.cos(azimuth_rad)
            dy = -math.sin(azimuth_rad)   # flip y axis for pixel space
 
            tip_x = int(cx_px + dx * arrow_len)
            tip_y = int(cy_px + dy * arrow_len)
 
            color = FACE_COLOR_BGR.get(face, (255, 255, 255))
 
            # dark shadow for visibility on bright backgrounds
            cv2.arrowedLine(annotated,
                            (cx_px, cy_px), (tip_x, tip_y),
                            (0, 0, 0), thickness + 2,
                            tipLength=0.25, line_type=cv2.LINE_AA)
            cv2.arrowedLine(annotated,
                            (cx_px, cy_px), (tip_x, tip_y),
                            color, thickness,
                            tipLength=0.25, line_type=cv2.LINE_AA)
 
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def get_tile_indices(xmin, ymin, xmax, ymax, matrix):
    tile_size = matrix.tilewidth * matrix.scaledenominator * 0.28e-3
    origin_x, origin_y = matrix.topleftcorner
    col_min = int((xmin - origin_x) // tile_size)
    col_max = int((xmax - origin_x) // tile_size)
    row_min = int((origin_y - ymax) // tile_size)
    row_max = int((origin_y - ymin) // tile_size)
    return col_min, col_max, row_min, row_max

def retrieve_satelite_image(top_left_corner, bottom_right_corner,progress_cb = None):
    wmts_url = (
        "https://cartografia.dgterritorio.gov.pt/ortos2018/service"
        "?service=WMTS&request=GetCapabilities"
    )
    wmts = WebMapTileService(wmts_url)

    xmin, ymax = top_left_corner
    xmax, ymin = bottom_right_corner

    layer          = "Ortos2018-RGB"
    tile_matrix_set = "PTTM_06"
    zoom_level     = "14"
    matrix   = wmts.tilematrixsets[tile_matrix_set].tilematrix[zoom_level]
    res      = matrix.scaledenominator * 0.28e-3
    tile_size_m = matrix.tilewidth * res

    col_min, col_max, row_min, row_max = get_tile_indices(xmin, ymin, xmax, ymax, matrix)
    n_rows = row_max + 1 - row_min
    n_cols = col_max + 1 - col_min
    total_tiles = n_rows * n_cols
    processed_blocks = np.empty((n_rows, n_cols), dtype=object)

    tile_cache_dir = os.path.join(CACHE_DIR, "tiles")
    os.makedirs(tile_cache_dir, exist_ok=True)

    done = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile_filename = os.path.join(tile_cache_dir, f"tile_{zoom_level}_{row}_{col}.png")
            if os.path.exists(tile_filename):
                img_array = np.array(Image.open(tile_filename).convert("RGB"))
            else:
                tile = wmts.gettile(
                    layer=layer, tilematrixset=tile_matrix_set,
                    tilematrix=zoom_level, row=row, column=col, format="image/png",
                )
                img = Image.open(io.BytesIO(tile.read())).convert("RGB")
                img.save(tile_filename) # Save to cache
                img_array = np.array(img)
            processed_blocks[row - row_min, col - col_min] = {"img": img_array}
            done += 1
            if progress_cb:
                progress_cb(done, total_tiles)

    block_h, block_w = processed_blocks[0, 0]["img"].shape[:2]
    stitched = np.zeros((n_rows * block_h, n_cols * block_w, 3), dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            img = processed_blocks[row, col]["img"]
            stitched[row*block_h:(row+1)*block_h, col*block_w:(col+1)*block_w] = img

    origin_x = matrix.topleftcorner[0] + col_min * tile_size_m
    origin_y = matrix.topleftcorner[1] - row_min * tile_size_m

    col_start = int(round((xmin - origin_x) / res))
    row_start = int(round((origin_y - ymax) / res))
    col_end   = int(round((xmax - origin_x) / res))
    row_end   = int(round((origin_y - ymin) / res))
    satellite_image = stitched[row_start:row_end, col_start:col_end, :]

    def conversion(x, y):
        return int((x - xmin) / res), int((ymax - y) / res)

    return satellite_image, res, conversion

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CachedModel:
    def __init__(self):
        self.model = YOLO(DEFAULT_MODEL_PATH)
        self.model.to("cpu")
        self.model.eval()

class RoofPlane:
    def __init__(self, face, probability, corners, normal, inclination, orientation):
        self.face = face;  self.probability = probability
        self.corners = corners;  self.normal = normal
        self.inclination = inclination;  self.orientation = orientation

class EstimatedBuilding:
    def __init__(self):
        self.box_coords_in_epsg_3763: list[float] = []
        self.planes_in_physical_dimensions: list[RoofPlane] = []
        self.raw_roof_data = []
        self.probabilities = []

    def convert_tensor_prediction_to_building(
        self, boxes_xywhn, roof_prediction, top_left_corner, res, image_shape
    ):
        self.raw_roof_data = roof_prediction
        img_h, img_w = image_shape[:2]
        xmin_map, ymax_map = top_left_corner
        cx_norm, cy_norm, w_norm, h_norm = boxes_xywhn
        cx_map = xmin_map + cx_norm * img_w * res
        cy_map = ymax_map - cy_norm * img_h * res
        w_map  = w_norm * img_w * res
        h_map  = h_norm * img_h * res

        self.box_coords_in_epsg_3763 = [
            cx_map - w_map/2, cy_map - h_map/2,
            cx_map + w_map/2, cy_map + h_map/2,
        ]
        probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
        self.probabilities.append(probh)
        self.probabilities.append(probtop)
        self.probabilities.append(probright)
        self.probabilities.append(probbottom)
        self.probabilities.append(probleft)
        ah, atop, aright, abottom, aleft = roof_prediction[5:10]
        inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
        ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]

        dx, dy = w_map/2, h_map/2
        p_tl = np.array([cx_map-dx, cy_map+dy, 0.])
        p_tr = np.array([cx_map+dx, cy_map+dy, 0.])
        p_br = np.array([cx_map+dx, cy_map-dy, 0.])
        p_bl = np.array([cx_map-dx, cy_map-dy, 0.])

        for face_name, prob, inc, ori, base_az, ps, pe in [
            ("top",    probtop,    inc_top,    ori_top,     math.pi/2,  p_tl, p_tr),
            ("right",  probright,  inc_right,  ori_right,   0.,         p_tr, p_br),
            ("bottom", probbottom, inc_bottom, ori_bottom, -math.pi/2,  p_br, p_bl),
            ("left",   probleft,   inc_left,   ori_left,    math.pi,    p_bl, p_tl),
        ]:
            if prob < 0.5: continue
            tilt    = inc * (math.pi/2)
            azimuth = (ori - 0.5) * (math.pi/2) + base_az
            normal  = np.array([math.sin(tilt)*math.cos(azimuth),
                                 math.sin(tilt)*math.sin(azimuth),
                                 math.cos(tilt)])
            normal /= np.linalg.norm(normal)
            edge_unit = (pe - ps) / np.linalg.norm(pe - ps)
            slope_vec = np.cross(edge_unit, normal)
            if slope_vec[2] > 0: slope_vec = -slope_vec
            sl = max(w_map, h_map) * 0.2
            self.planes_in_physical_dimensions.append(RoofPlane(
                face=face_name, probability=float(prob),
                corners=[ps, pe, pe+slope_vec*sl, ps+slope_vec*sl],
                normal=normal, inclination=float(inc), orientation=float(ori),
            ))

def _iter_tiles(image, top_left_corner, res, tile_px, overlap=0.2):
    """
    Yield (tile_img, tile_top_left_corner) for every window position.
 
    Parameters
    ----------
    image           : np.ndarray  full satellite image (H, W, 3)
    top_left_corner : (x, y)      EPSG:3763 coords of the image top-left pixel
    res             : float        metres per pixel
    tile_px         : int          tile side in pixels (match your model input, e.g. 640)
    overlap         : float        fractional overlap between adjacent tiles (0.0â€“0.5)
    """
    img_h, img_w = image.shape[:2]
    stride = int(tile_px * (1.0 - overlap))   # pixels between tile starts
    tl_x, tl_y = top_left_corner
 
    row_start = 0
    while True:
        row_end = row_start + tile_px
        # Clamp so we never go out of bounds; shift start back instead
        if row_end > img_h:
            row_start = max(0, img_h - tile_px)
            row_end   = img_h
 
        col_start = 0
        while True:
            col_end = col_start + tile_px
            if col_end > img_w:
                col_start = max(0, img_w - tile_px)
                col_end   = img_w
 
            tile_img = image[row_start:row_end, col_start:col_end]
 
            # EPSG:3763 top-left of THIS tile
            tile_tl_x = tl_x + col_start * res
            tile_tl_y = tl_y - row_start * res   # y decreases downward
 
            yield tile_img, (tile_tl_x, tile_tl_y)
 
            if col_end == img_w:
                break
            col_start += stride
 
        if row_end == img_h:
            break
        row_start += stride
 
 
def _box_iou(box_a, box_b):
    """
    box = [xmin, ymin, xmax, ymax] in EPSG:3763 metres.
    Returns IoU scalar.
    """
    # should this be with the rotated bounding boxes? My spider senses tell me yes
    # but it takes time, lets say TODO 
    ix = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    iy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    inter = ix * iy
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
 
 
def _nms_predictions(predictions, iou_threshold=0.5):
    """
    This function exists because OVEN cannot parse a large satellite image, hence
    the results need to be stiched together. The question arises, how can we do this?
    We basically check boxes that overall and eliminate the ones that do. 
    The surviver is the one with the highest degree of confidence in its predictions.
    """
    if not predictions:
        return []
 
    def _score(b):
        if len(b.probabilities) == 0:
            return 0.0
        return max(b.probabilities)
 
    ranked = sorted(predictions, key=_score, reverse=True)
    kept   = []
 
    for candidate in ranked:
        box_c = candidate.box_coords_in_epsg_3763
        suppressed = False
        for accepted in kept:
            box_a = accepted.box_coords_in_epsg_3763
            if _box_iou(box_c, box_a) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
 
    return kept
  
def retrieve_prediction_list(
    satellite_image,
    top_left_corner,
    res,
    building_threshold,
    overlap_threshold,
    cached_model,
    tile_overlap=0.2,
    nms_iou=0.5,
):
    """
    Tiled inference replacement.
 
    Extra parameters vs. original
    ------------------------------
    tile_px      : model input size in pixels (default 640)
    tile_overlap : fractional overlap between adjacent tiles (default 0.2 = 20 %)
    nms_iou      : IoU threshold for cross-tile duplicate suppression (default 0.5)
    """
    img_h, img_w = satellite_image.shape[:2]
    all_predictions = []
    tile_idx = 0
 
    for tile_img, tile_tl in _iter_tiles(satellite_image, top_left_corner, res, cached_model.model.model.args.get('imgsz', 640), tile_overlap):
        tile_idx += 1
        print(f"  ðŸ”² Tile {tile_idx} â€” tl=({tile_tl[0]:.0f}, {tile_tl[1]:.0f}), "
              f"size={tile_img.shape[1]}Ã—{tile_img.shape[0]}")
 
        results = cached_model.model(
            tile_img,
            conf=building_threshold,
            iou=overlap_threshold,   # within-tile NMS, same as before
        )[0]
 
        for i in range(results.boxes.shape[0]):
            b = EstimatedBuilding()
            b.convert_tensor_prediction_to_building(
                boxes_xywhn=np.array(results.boxes[i].xywhn[0]),
                roof_prediction=np.array(sigmoid(results.roof.data[i, :])),
                top_left_corner=tile_tl,
                res=res,
                image_shape=tile_img.shape,
            )
            all_predictions.append(b)
 
    print(f"  ðŸ“¦ {len(all_predictions)} raw detections across {tile_idx} tile(s)")
    final = _nms_predictions(all_predictions, iou_threshold=nms_iou)
    print(f"  âœ… {len(final)} after global NMS (iouâ‰¥{nms_iou})")
    return final

def get_osm_buildings(top_left_corner, bottom_right_corner):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s, n = min(tl_lat, br_lat), max(tl_lat, br_lat)
    w, e = min(tl_lon, br_lon), max(tl_lon, br_lon)
    q = f"""[out:json];(way["building"]({s},{w},{n},{e});
    relation["building"]({s},{w},{n},{e}););out body;>;out skel qt;"""
    r = requests.post("https://overpass-api.de/api/interpreter", data=q)
    r.raise_for_status()
    data  = r.json()
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in data["elements"] if el["type"]=="node"}
    features = []
    for el in data["elements"]:
        if el["type"] != "way": continue
        coords = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(coords) < 3: continue
        features.append({"type":"Feature",
                          "properties":{"osm_id":el["id"], **el.get("tags",{})},
                          "geometry":{"type":"Polygon","coordinates":[coords]}})
    return {"type":"FeatureCollection","features":features}

def get_osm_buildings_cached(top_left, bottom_right):
    '''
    ok I just need this because when doing tests I don't want to constantly download the tiles
    so first I check if the OSM buildings have been cached, and then I do things
    '''
    osm_cache_dir = os.path.join(CACHE_DIR, "osm")
    os.makedirs(osm_cache_dir, exist_ok=True)
    
    # Create a unique ID based on the coordinates
    coord_str = f"{top_left}_{bottom_right}"
    cache_key = hashlib.md5(coord_str.encode()).hexdigest()
    cache_path = os.path.join(osm_cache_dir, f"{cache_key}.json")

    if os.path.exists(cache_path):
        print(f"ðŸ“¦ Loading OSM data from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    # If not cached, fetch it
    data = get_osm_buildings(top_left, bottom_right)
    
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data


def build_cea_ordered_buildings_geojson(polygon_ring_lon_lat, zone_shp_path=None):
    """
    Load authoritative CEA building footprints from scenario `zone.shp`,
    filter by polygon intersection, and expose `properties.cea_name`
    from zone `name` values without any renaming.
    """
    if len(polygon_ring_lon_lat) < 4:
        raise ValueError("Polygon must include at least 4 points (closed ring).")

    ring = [[float(lon), float(lat)] for lon, lat in polygon_ring_lon_lat]
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GeoPandas is required to read scenario zone.shp footprints."
        ) from exc

    from shapely.geometry import Polygon as ShapelyPolygon

    resolved_zone_shp_path = str(zone_shp_path or ZONE_SHP_PATH).strip()
    if not resolved_zone_shp_path:
        raise ValueError("A zone.shp path is required.")

    if not os.path.exists(resolved_zone_shp_path):
        raise FileNotFoundError(f"Scenario zone.shp not found: {resolved_zone_shp_path}")

    zone_df = gpd.read_file(resolved_zone_shp_path)
    if "name" not in zone_df.columns:
        raise ValueError("Scenario zone.shp is missing required `name` column.")
    if zone_df.crs is None:
        raise ValueError("Scenario zone.shp has no CRS; cannot transform to EPSG:4326.")

    zone_df = zone_df.to_crs("EPSG:4326")
    zone_df = zone_df[
        zone_df.geometry.notnull()
        & (~zone_df.geometry.is_empty)
        & zone_df.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    selected_polygon = ShapelyPolygon(ring)
    zone_df = zone_df[zone_df.intersects(selected_polygon)].copy()

    zone_df["name"] = zone_df["name"].astype(str).str.strip()
    zone_df = zone_df[(zone_df["name"] != "") & (zone_df["name"].str.lower() != "nan")].copy()
    zone_df = zone_df.dissolve(by="name", as_index=False)
    zone_df = zone_df.sort_values("name").reset_index(drop=True)

    features = []
    for _, row in zone_df.iterrows():
        cea_name = str(row.get("name", "")).strip()

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if geom.geom_type == "MultiPolygon":
            polygon = max(list(geom.geoms), key=lambda g: g.area)
        elif geom.geom_type == "Polygon":
            polygon = geom
        else:
            continue

        if "building" in zone_df.columns:
            building_type = str(row.get("building", ""))
        elif "use_type1" in zone_df.columns:
            building_type = str(row.get("use_type1", ""))
        else:
            building_type = ""

        coords = [[float(x), float(y)] for x, y, *_ in polygon.exterior.coords]
        features.append(
            {
                "type": "Feature",
                "properties": {"cea_name": cea_name, "building": building_type},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )

    if not features:
        raise ValueError("No zone.shp buildings intersect the selected polygon.")

    first_five = [f["properties"]["cea_name"] for f in features[:5]]
    print(f"[info] zone.shp buildings selected: {len(features)}; first IDs: {first_five}")

    return {"type": "FeatureCollection", "features": features}

import numpy as np
import math
from pyproj import Transformer
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import cv2

def planenormal(face_id, inc, ori,axis_aligned_bounding_box_rotation):
    # user of this code! be aware, it took me a billion years to catch this discusting error
    # caused by the axis_aligned_bounding_box_rotation. Appreciate my sacrifice for your benefit!
    if face_id == "T":
        base_ori_rad = math.pi / 2
    elif face_id == "B":
        base_ori_rad = -math.pi / 2 
    elif face_id == "R":
        base_ori_rad = 0 
    elif face_id == "L":
        base_ori_rad = math.pi 
    elif face_id == "H":
        return [0, 0, 1],0.0
    else:
        return None,0.0
    tilt = inc * (math.pi / 2)
    # v3 modelling choice: force roof-face orientation to the neutral midpoint.
    ori = 0.5
    azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad
    nz = math.cos(tilt)
    nx = math.sin(tilt) * math.cos(azimuth)
    ny = math.sin(tilt) * math.sin(azimuth)
    return [nx, ny, nz],azimuth
 
def split_with_lines(corners, lines):
    """Split a polygon defined by corners using a list of lines"""
    polys = [Polygon(corners)]
    for line in lines:
        new_polys = []
        splitter = LineString(line)
 
        for poly in polys:
            result = split(poly, splitter)
 
            if len(result.geoms) > 1:
                new_polys.extend(result.geoms)
            else:
                new_polys.append(poly)
        polys = new_polys
    return [np.array(p.exterior.coords[:-1]) for p in polys]
 
def determine_face_for_polygon(poly_centroid, corners, code):
    """
    Determine which face a polygon belongs to based on its centroid position.
    Returns the face identifier (e.g., 'T', 'R', 'B', 'L', 'H')
    
    corners: [[x_min_l, y_min_l], [x_max_l, y_min_l], [x_max_l, y_max_l], [x_min_l, y_max_l]]
    """
    x_min_l, y_min_l = corners[0]
    x_max_l, y_max_l = corners[2]
    cx, cy = poly_centroid
    
    # For cases with H (horizontal/flat), check if centroid is near center
    if 'H' in code:
        # Define a central region
        center_threshold = 0.3  # 30% of dimension from center
        x_center_min = x_min_l + (x_max_l - x_min_l) * (0.5 - center_threshold)
        x_center_max = x_min_l + (x_max_l - x_min_l) * (0.5 + center_threshold)
        y_center_min = y_min_l + (y_max_l - y_min_l) * (0.5 - center_threshold)
        y_center_max = y_min_l + (y_max_l - y_min_l) * (0.5 + center_threshold)
        
        if (x_center_min <= cx <= x_center_max and y_center_min <= cy <= y_center_max):
            return 'H'
    
    # Otherwise, determine by position relative to edges
    # Calculate distances to each edge
    dist_to_top = abs(cy - y_max_l)
    dist_to_bottom = abs(cy - y_min_l)
    dist_to_right = abs(cx - x_max_l)
    dist_to_left = abs(cx - x_min_l)
    
    # Find the nearest edge
    distances = {
        'T': dist_to_top,
        'B': dist_to_bottom,
        'R': dist_to_right,
        'L': dist_to_left
    }
    
    # Only consider faces that are in the code
    valid_distances = {face: dist for face, dist in distances.items() if face in code}
    
    if valid_distances:
        nearest_face = min(valid_distances, key=valid_distances.get)
        return nearest_face
    
    # Fallback: use quadrant-based logic
    if cx >= 0 and cy >= 0:
        return 'T' if 'T' in code else ('R' if 'R' in code else 'H')
    elif cx >= 0 and cy < 0:
        return 'R' if 'R' in code else ('B' if 'B' in code else 'H')
    elif cx < 0 and cy < 0:
        return 'B' if 'B' in code else ('L' if 'L' in code else 'H')
    else:  # cx < 0 and cy >= 0
        return 'L' if 'L' in code else ('T' if 'T' in code else 'H')
 
def rooftile(corners, plane_point, plane_data, height=1000.0):
    """
    This function computes the intersection of the corners of 
    the section of the building reserved for this direction
    with the roof predicted by the OVEN model
    """
    import pyvista as pv
    plane_normal,azymuth = plane_data 
    pts = np.array(corners)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    n_pts = len(pts)
    faces = [n_pts] + list(range(n_pts))
    intersection_polygon = pv.PolyData(pts, faces=faces).extrude((0, 0, height), capping=True).slice(normal=plane_normal, origin=plane_point)
    
    return intersection_polygon,azymuth
 
def topology_converter_mine(roof_prediction, osm_building):
    """
    Convert roof predictions to topology with plane intersections.
    
    Returns:
        outline: Building outline in 3D
        rect: Oriented bounding box (center, size, angle)
        lines_world: Dividing lines in world coordinates
        code: Roof topology code (e.g., "HTRBL")
        face_data: Dictionary of face data including intersections
    """
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    ring = osm_building["geometry"]["coordinates"][0]
    xs, ys = zip(*[tr.transform(lon, lat) for lon, lat in ring])
 
    outline = np.concatenate((np.array(xs), np.array(ys), np.zeros_like(np.array(xs))), axis=0)
    outline = outline.reshape(3, -1)
 
    probh, probtop, probright, probbottom, probleft = roof_prediction[0:5]
    inc_top, inc_right, inc_bottom, inc_left = roof_prediction[10:14]
    ori_top, ori_right, ori_bottom, ori_left = roof_prediction[14:18]
 
    points = outline[:2, :].T.astype(np.float32)
    rect = cv2.minAreaRect(points)
    center, (width, height), angle = rect

    if angle < -45:
        angle = angle + 90
        tmp = height
        height = width
        width = tmp

    x_min_l = -width / 2.0
    x_max_l = width / 2.0
    y_min_l = -height / 2.0
    y_max_l = height / 2.0
 
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
 
    global_probs = {"T": probtop, "R": probright, "B": probbottom, "L": probleft}
    global_inc = {"T": inc_top, "R": inc_right, "B": inc_bottom, "L": inc_left}
    global_ori = {"T": ori_top, "R": ori_right, "B": ori_bottom, "L": ori_left}

    face_data = {
        "T": {"active": global_probs["T"] > 0.5, "inclination": global_inc["T"], "orientation": global_ori["T"]},
        "R": {"active": global_probs["R"] > 0.5, "inclination": global_inc["R"], "orientation": global_ori["R"]},
        "B": {"active": global_probs["B"] > 0.5, "inclination": global_inc["B"], "orientation": global_ori["B"]},
        "L": {"active": global_probs["L"] > 0.5, "inclination": global_inc["L"], "orientation": global_ori["L"]},
        "H": {"active": probh > 0.5, "inclination": 0.0, "orientation": 0.0},
    }
 
    code = ""

    ## Ok important, we need to deal with the case where the probability of none is higher than 50% 
    # but one must be there because a building was identified. 
    # What I suggest is the following: We select the highest probability, and check minus 0.05 percent bellow that maximum
    maximumprob = roof_prediction[0:5].max()

    maximumprob =  0.5*0.8 if maximumprob > 0.5 else maximumprob*0.80   
    if probh > maximumprob: code += "H"
    if global_probs["T"] > maximumprob: code += "T"
    if global_probs["R"] > maximumprob: code += "R"
    if global_probs["B"] > maximumprob: code += "B"
    if global_probs["L"] > maximumprob: code += "L"

    if len(code) == 0: #this is a sanity check
        raise NameError('The code must never be empty. If a building exists, then at least one rooftop is present')

    lines = []
    corners = np.array([
        [x_min_l, y_min_l],
        [x_max_l, y_min_l],
        [x_max_l, y_max_l],
        [x_min_l, y_max_l],
    ])
    
    intersections = {}
    
    base_height = 10.0
    
    match code:
        case "L":
            normal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            intersections["L"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "B":
            normal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            intersections["B"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "R":
            normal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            intersections["R"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "T":
            normal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            intersections["T"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "H":
            normal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            intersections["H"] = rooftile(corners, np.array([0, 0, base_height]), normal)
            
        case "BL": #corrected
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "RL": #corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "RB":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "HL":
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "HB":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "TL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "L":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Lnormal)
                    
        case "TB":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "B":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Bnormal)
                    
        case "TR":
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                elif face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                    
        case "HR":
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "R":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Rnormal)
                    
        case "HT":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                if face == "H":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Hnormal)
                elif face == "T":
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), Tnormal)
                    
        case "RBL":
            lines.append([[0, y_min_l], [x_min_l, y_max_l]])
            lines.append([[0, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TBL":
            lines.append([[x_max_l, 0], [x_min_l, y_min_l]])
            lines.append([[x_max_l, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRL":
            lines.append([[0, y_max_l], [x_min_l, y_min_l]])
            lines.append([[0, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRB":
            lines.append([[x_min_l, 0], [x_max_l, y_min_l]])
            lines.append([[x_min_l, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTB":
            lines.append([[0, y_min_l], [0, y_max_l]])
            lines.append([[0, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HBL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[0, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRL":
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            lines.append([[0, 0], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRB":
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTR":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "TRBL":
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"T": Tnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HRBL":
            lines.append([[0, 0], [x_min_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTBL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRL":
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRB":
            lines.append([[0, 0], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "B": Bnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
                    
        case "HTRBL":
            # Complex hip roof with ridge
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [0.5*x_min_l, 0.5*y_max_l]])
            lines.append([[0.5*x_max_l, 0.5*y_min_l], [0.5*x_max_l, 0.5*y_max_l]])
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [0.5*x_max_l, 0.5*y_min_l]])
            lines.append([[0.5*x_min_l, 0.5*y_max_l], [0.5*x_max_l, 0.5*y_max_l]])
            lines.append([[0.5*x_min_l, 0.5*y_min_l], [x_min_l, y_min_l]])
            lines.append([[0.5*x_max_l, 0.5*y_min_l], [x_max_l, y_min_l]])
            lines.append([[0.5*x_min_l, 0.5*y_max_l], [x_min_l, y_max_l]])
            lines.append([[0.5*x_max_l, 0.5*y_max_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            Hnormal = planenormal("H", face_data["H"]["inclination"], face_data["H"]["orientation"],theta)
            Tnormal = planenormal("T", face_data["T"]["inclination"], face_data["T"]["orientation"],theta)
            Rnormal = planenormal("R", face_data["R"]["inclination"], face_data["R"]["orientation"],theta)
            Bnormal = planenormal("B", face_data["B"]["inclination"], face_data["B"]["orientation"],theta)
            Lnormal = planenormal("L", face_data["L"]["inclination"], face_data["L"]["orientation"],theta)
            
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, code)
                normal = {"H": Hnormal, "T": Tnormal, "R": Rnormal, "B": Bnormal, "L": Lnormal}.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal)
    
    # Convert lines to world coordinates
    cx, cy = center
    for face in intersections:
        # Double the count of every intersection
        intersection,azymuth = intersections[face]
        intersection.rotate_z(angle, inplace=True)
        intersection.translate([cx, cy, 0], inplace=True)

    # Store intersections in face_data
    face_data["intersections"] = intersections

    def local_to_world(pt):
        xl, yl = pt
        xw = cx + xl * cos_t - yl * sin_t
        yw = cy + xl * sin_t + yl * cos_t
        return [xw, yw]
 
    lines_world = [[local_to_world(p0), local_to_world(p1)] for p0, p1 in lines]
 
    return outline, rect, lines_world, code, face_data

from pyproj import Transformer
import numpy as np

def compute_iou_matrix(osm_buildings, predictions):
    osm_boxes = []
    _TR_4326_TO_3763 = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    for feat in osm_buildings:
        xs, ys = zip(*[_TR_4326_TO_3763.transform(lon, lat)
                    for lon, lat in feat["geometry"]["coordinates"][0]])
        osm_boxes.append([min(xs), min(ys), max(xs), max(ys)])

    iou_mat = np.zeros((len(osm_boxes), len(predictions)))
    for i, osm_box in enumerate(osm_boxes):
        for j, pred in enumerate(predictions):
            pred_box = pred.box_coords_in_epsg_3763
            ix0 = max(osm_box[0], pred_box[0])
            iy0 = max(osm_box[1], pred_box[1])
            ix1 = min(osm_box[2], pred_box[2])
            iy1 = min(osm_box[3], pred_box[3])
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            area_osm  = (osm_box[2] - osm_box[0]) * (osm_box[3] - osm_box[1])
            area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            union = area_osm + area_pred - inter
            iou_mat[i, j] = inter / union if union > 0 else 0.0
    return iou_mat


def _feature_building_id(feature, fallback="?"):
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
    return str(props.get("cea_name") or props.get("osm_id") or fallback)


def _entry_building_id(entry, fallback="?"):
    return str(
        entry.get("building_id")
        or entry.get("cea_name")
        or entry.get("osm_id")
        or fallback
    )


def _ordered_ring_from_points(points_3d):
    if points_3d.shape[0] < 3:
        return None

    pts = np.unique(np.round(points_3d, 6), axis=0)
    if pts.shape[0] < 3:
        return None

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis_u = vh[0]
    normal = vh[2]
    axis_v = np.cross(normal, axis_u)

    norm_u = np.linalg.norm(axis_u)
    norm_v = np.linalg.norm(axis_v)
    if norm_u == 0.0 or norm_v == 0.0:
        return None

    axis_u = axis_u / norm_u
    axis_v = axis_v / norm_v

    uv = np.column_stack(
        [
            np.dot(centered, axis_u),
            np.dot(centered, axis_v),
        ]
    )
    angles = np.arctan2(uv[:, 1], uv[:, 0])
    order = np.argsort(angles)
    ordered = pts[order]

    if ordered.shape[0] < 3:
        return None

    ring = ordered.tolist()
    ring.append(ordered[0].tolist())
    return ring


def export_roof_surfaces_geojson(
    matched_data,
    output_path="roof_surfaces.geojson",
    source_crs="EPSG:3763",
    target_crs="EPSG:32629",
):
    """
    Export roof surfaces from face intersections to GeoJSON.
    One feature is written per (building, face) surface.
    """
    tr = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    features = []

    for entry in matched_data:
        building_id = _entry_building_id(entry, fallback="unknown")
        intersections = entry.get("face_data", {}).get("intersections", {})
        roof_counter = 1

        for _, tile_obj in intersections.items():
            tile_mesh = tile_obj
            if isinstance(tile_obj, (tuple, list)) and len(tile_obj) > 0:
                tile_mesh = tile_obj[0]

            if tile_mesh is None or not hasattr(tile_mesh, "n_points") or tile_mesh.n_points < 3:
                continue

            points = np.asarray(tile_mesh.points)
            if points.ndim != 2 or points.shape[1] < 3:
                continue

            ring_3d_src = _ordered_ring_from_points(points)
            if ring_3d_src is None or len(ring_3d_src) < 4:
                continue

            ring_3d_dst = []
            for x, y, z in ring_3d_src:
                x_t, y_t = tr.transform(float(x), float(y))
                ring_3d_dst.append([x_t, y_t, float(z)])

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "building": building_id,
                        "roof_id": str(roof_counter),
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [ring_3d_dst],
                    },
                }
            )
            roof_counter += 1

    geojson = {
        "type": "FeatureCollection",
        "name": "roof_surfaces",
        "crs": {
            "type": "name",
            "properties": {"name": f"urn:ogc:def:crs:{target_crs.replace(':', '::')}"},
        },
        "features": features,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"[ok] Roof surfaces exported to: {output_path} ({len(features)} feature(s))")
    return geojson

model = CachedModel()

import pydeck as pdk
from pyproj import Transformer

# Ok to convert quickly we allocate one transformer that is repeatedly used
_tr_3763_to_4326 = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
 
def _to_lonlat(x, y):
    lon, lat = _tr_3763_to_4326.transform(x, y)
    return float(lon), float(lat)
 
 
# â”€â”€ layer builders â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _make_geojson_layer(geojson_fc) -> pdk.Layer:
    """
    OSM building footprints â†’ GeoJsonLayer (extruded, flat-roofed reference).
    The `geojson_fc` is the FeatureCollection returned by get_osm_buildings_cached().
    Coordinates are already in EPSG:4326 (lon/lat) as required by deck.gl.
    """
    return pdk.Layer(
        "GeoJsonLayer",
        id="zone-footprints",
        data=geojson_fc,                    # pass the dict directly
        opacity=0.4,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation=8,                    # flat 8 m placeholder; swap for real height tag:
                                            # "properties.height" or "properties['building:levels'] * 3"
        get_fill_color=[100, 180, 255, 120],
        get_line_color=[255, 255, 255, 200],
        pickable=True,
        auto_highlight=True,
    )
 
 
def _make_osm_outline_layer(geojson_fc) -> pdk.Layer:
    """
    OSM building footprints â†’ flat PolygonLayer drawn at z=0.
    This is the 'street outline' â€” always visible regardless of pitch,
    because it sits on the ground and is never hidden by the extruded boxes.
    Coordinates are already in EPSG:4326 as returned by get_osm_buildings_cached().
    """
    records = []
    for feat in geojson_fc.get("features", []):
        coords = feat["geometry"]["coordinates"][0]   # outer ring, list of [lon, lat]
        records.append({
            "polygon": [[lon, lat] for lon, lat in coords],
            "building_id": _feature_building_id(feat),
        })
 
    return pdk.Layer(
        "PolygonLayer",
        id="zone-outlines",
        data=records,
        get_polygon="polygon",
        get_fill_color=[0, 0, 0, 0],            # fully transparent fill
        get_line_color=[0, 220, 255, 255],       # bright cyan outline
        stroked=True,
        filled=False,
        extruded=False,
        line_width_min_pixels=2,
        pickable=True,
    )
 
 
def _make_bounding_box_layer(matched_data) -> pdk.Layer:
    """
    cv2.minAreaRect oriented bounding boxes â†’ PolygonLayer.
    Each entry in matched_data must have "roof_planes" (the cv2 rotated-rect tuple).
    """
    import cv2
    records = []
    for entry in matched_data:
        rect = entry.get("roof_planes")
        if rect is None:
            continue
        corners_3763 = cv2.boxPoints(rect).astype(float)     # (4,2) in EPSG:3763
        polygon_lonlat = []
        for x, y in corners_3763:
            lon, lat = _to_lonlat(x, y)
            polygon_lonlat.append([lon, lat])
        polygon_lonlat.append(polygon_lonlat[0])              # close ring
        records.append({
            "polygon": polygon_lonlat,
            "building_id": _entry_building_id(entry),
            "code":    entry.get("code", ""),
        })
 
    return pdk.Layer(
        "PolygonLayer",
        id="roof-bounds",
        data=records,
        get_polygon="polygon",
        get_fill_color=[0, 255, 136, 50],   # translucent green
        get_line_color=[0, 204, 102, 220],
        stroked=True,
        filled=True,
        extruded=False,
        line_width_min_pixels=1,
        pickable=True,
    )
 
 
def _make_roof_face_layer(matched_data) -> pdk.Layer:
    """
    PyVista face intersection polygons â†’ PolygonLayer with elevation (3-D).
 
    Each intersection polygon is already a PyVista PolyData whose `.points`
    are in EPSG:3763 (x, y, z_metres).  We convert x/y to lon/lat and keep z
    as the elevation so deck.gl renders the sloped faces in true 3-D.
 
    Color scheme:
        T (top / flat)  â†’ warm orange
        H (hip)         â†’ warm orange
        R / L / B       â†’ blue-grey slope faces
    """
    _face_colors = {
        "T": [255, 140,  40, 220],
        "H": [255, 180,  80, 220],
        "R": [ 80, 140, 220, 200],
        "L": [ 80, 140, 220, 200],
        "B": [ 80, 140, 220, 200],
    }
    _default_color = [160, 160, 160, 180]
 
    records = []
    for entry in matched_data:
        face_data   = entry.get("face_data", {})
        intersections = face_data.get("intersections", {}) if face_data else {}
        building_id = _entry_building_id(entry)
        code = entry.get("code", {})
        for face_name, plane in intersections.items():
            if plane is None:
                continue
            points,azymuth = plane 
            raw_pts = np.asarray(points.points)
            if raw_pts.ndim != 2 or raw_pts.shape[1] < 3:
                continue
            ring_3d = _ordered_ring_from_points(raw_pts)
            if ring_3d is None:
                continue
            polygon_3d = []
            for v in ring_3d:
                lon, lat = _to_lonlat(v[0], v[1])
                polygon_3d.append([lon, lat, float(v[2])])  # z kept as metres
 
            records.append({
                "polygon":   polygon_3d,
                "azymuth": round(math.degrees(azymuth), 1),   # centroid height for tooltip
                "face":      face_name,
                "building_id": building_id,
                "code" :   code ,
                "color":     _face_colors.get(face_name, _default_color),
            })
 
    return pdk.Layer(
        "PolygonLayer",
        id="roof-surfaces",
        data=records,
        get_polygon="polygon",
        get_fill_color="color",
        get_elevation=0,        # elevation is baked into the polygon z-coords
        extruded=False,         # False because z is already in the vertices
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        get_line_color=[255, 255, 255, 180],
        pickable=True,
        auto_highlight=True,
    )
 
 
def _make_topology_line_layer(matched_data) -> pdk.Layer:
    """
    Roof ridge / valley topology lines â†’ PathLayer.
    Each entry has "lines_world": list of [[x0,y0], [x1,y1]] in EPSG:3763.
    """
    records = []
    for entry in matched_data:
        building_id = _entry_building_id(entry)
        lines_world = entry.get("lines_world", [])
        for seg in lines_world:
            p0, p1 = seg
            lon0, lat0 = _to_lonlat(p0[0], p0[1])
            lon1, lat1 = _to_lonlat(p1[0], p1[1])
            records.append({
                "path":   [[lon0, lat0], [lon1, lat1]],
                "building_id": building_id,
            })
 
    return pdk.Layer(
        "PathLayer",
        id="roof-lines",
        data=records,
        get_path="path",
        get_color=[255, 255, 255, 230],
        get_width=0.3,          # metres
        width_min_pixels=2,
        pickable=True,
    )
 
 
# â”€â”€ view-state helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
 
def _center_view(matched_data, geojson_fc):
    """Compute a sensible initial ViewState from the data bounding box."""
    lons, lats = [], []
 
    # pull coords from matched_data bounding boxes
    import cv2
    for entry in matched_data:
        rect = entry.get("roof_planes")
        if rect is None:
            continue
        for x, y in cv2.boxPoints(rect).astype(float):
            lon, lat = _to_lonlat(x, y)
            lons.append(lon); lats.append(lat)
 
    # fallback: use OSM footprint centroids
    if not lons:
        for feat in geojson_fc.get("features", []):
            coords = feat["geometry"]["coordinates"][0]
            for lon, lat in coords:
                lons.append(lon); lats.append(lat)
 
    if not lons:
        return pdk.ViewState(latitude=38.712, longitude=-9.142, zoom=18, pitch=45)
 
    return pdk.ViewState(
        latitude=sum(lats) / len(lats),
        longitude=sum(lons) / len(lons),
        zoom=18,
        pitch=58,
        bearing=-12,
        max_zoom=22,
    )


def _inject_layer_toggle_panel(html_str, layer_items):
    if not layer_items:
        return html_str

    panel_data = json.dumps(layer_items, ensure_ascii=True)
    control_script = f"""
<script>
(function () {{
  const layerItems = {panel_data};
  if (!Array.isArray(layerItems) || layerItems.length === 0) return;

  function resolveDeckInstance() {{
    if (typeof deckInstance === "undefined" || !deckInstance) return null;
    if (typeof deckInstance.setProps === "function") return deckInstance;
    if (deckInstance.deck && typeof deckInstance.deck.setProps === "function") return deckInstance.deck;
    return null;
  }}

  const deckObj = resolveDeckInstance();
  if (!deckObj || !deckObj.props || !Array.isArray(deckObj.props.layers)) return;

  const baseLayers = deckObj.props.layers.slice();
  const visibility = {{}};
  layerItems.forEach((item) => {{
    visibility[item.id] = item.visible !== false;
  }});

  function applyVisibility() {{
    const updatedLayers = baseLayers.map((layer) => {{
      if (!layer || !layer.id || !(layer.id in visibility)) return layer;
      if (typeof layer.clone === "function") {{
        return layer.clone({{ visible: visibility[layer.id] }});
      }}
      return layer;
    }});
    deckObj.setProps({{ layers: updatedLayers }});
  }}

  const panel = document.createElement("div");
  panel.id = "roof-layer-panel";
  panel.style.position = "absolute";
  panel.style.top = "12px";
  panel.style.right = "12px";
  panel.style.zIndex = "10";
  panel.style.background = "rgba(20, 25, 32, 0.88)";
  panel.style.color = "#ffffff";
  panel.style.padding = "10px 12px";
  panel.style.border = "1px solid rgba(255, 255, 255, 0.2)";
  panel.style.borderRadius = "8px";
  panel.style.fontFamily = "Segoe UI, sans-serif";
  panel.style.fontSize = "13px";
  panel.style.minWidth = "210px";

  const title = document.createElement("div");
  title.textContent = "Map Layers";
  title.style.fontWeight = "600";
  title.style.marginBottom = "8px";
  panel.appendChild(title);

  layerItems.forEach((item) => {{
    const row = document.createElement("label");
    row.style.display = "flex";
    row.style.alignItems = "center";
    row.style.gap = "8px";
    row.style.marginBottom = "6px";
    row.style.cursor = "pointer";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = visibility[item.id];
    checkbox.style.margin = "0";
    checkbox.addEventListener("change", () => {{
      visibility[item.id] = checkbox.checked;
      applyVisibility();
    }});

    const text = document.createElement("span");
    text.textContent = item.label;

    row.appendChild(checkbox);
    row.appendChild(text);
    panel.appendChild(row);
  }});

  document.body.appendChild(panel);
  applyVisibility();
}})();
</script>
"""

    if "</body>" in html_str:
        return html_str.replace("</body>", f"{control_script}\n  </body>", 1)
    return f"{html_str}\n{control_script}"
 
  
def render_pydeck(
    geojson_fc,
    matched_data,
    out_path="3d_roofs.html",
    map_style="dark",   # "dark" | "light" | "satellite" | "road"
):
    """
    Build a pydeck Deck with layered 3-D output and write it to `out_path`.
 
    Layers (bottom â†’ top):
        1. GeoJsonLayer   â€“ OSM footprints (extruded reference boxes)
        2. PolygonLayer   â€“ Oriented min-bounding boxes
        3. PolygonLayer   â€“ Roof face intersections (PyVista planes)
        4. PathLayer      â€“ Topology ridge/valley lines
    """
    _map_styles = {
        "dark":  pdk.map_styles.DARK,
        "light": pdk.map_styles.LIGHT,
        "road":  pdk.map_styles.ROAD,
    }
 
    layers = [
        _make_geojson_layer(geojson_fc),        # 1. extruded 3D reference boxes
        _make_osm_outline_layer(geojson_fc),    # 2. flat cyan footprint outline (street level)
        _make_bounding_box_layer(matched_data), # 3. oriented min-bounding boxes
        _make_roof_face_layer(matched_data),    # 4. roof face intersections (PyVista planes)
        _make_topology_line_layer(matched_data),# 5. topology ridge/valley lines
    ]
    layer_items = [
        {"id": "zone-footprints", "label": "Zone Footprints (3-D ref)", "visible": True},
        {"id": "zone-outlines", "label": "Zone Outlines", "visible": True},
        {"id": "roof-bounds", "label": "Min Bounding Boxes", "visible": True},
        {"id": "roof-surfaces", "label": "Roof Surfaces (3-D)", "visible": True},
        {"id": "roof-lines", "label": "Topology Lines", "visible": True},
    ]
 
    view_state = _center_view(matched_data, geojson_fc)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style=_map_styles.get(map_style, pdk.map_styles.ROAD),
        tooltip={
            "html": (
                "<b>Building {building_id}</b><br/>"
                "Face: {face} | Code: {code}<br/>"
                "Azymuth: {azymuth} Âº"
            ),
            "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
        },
    )
 
    html_str = deck.to_html(as_string=True)
    html_str = _inject_layer_toggle_panel(html_str, layer_items)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(html_str)
    print(f"[ok] pydeck map saved to: {out_path}")
    return html_str


def build_map_html(osm_geojson, predicted_buildings, satellite_image,
                   top_left_corner, bottom_right_corner, matched_data=None):
    tr = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)

    def box_poly(box):
        x0,y0,x1,y1 = box
        return [tr.transform(x,y)[::-1] for x,y in [(x0,y1),(x1,y1),(x1,y0),(x0,y0),(x0,y1)]]

    lats, lons = [], []
    for f in osm_geojson["features"]:
        for lon, lat in f["geometry"]["coordinates"][0]:
            lats.append(lat); lons.append(lon)
    center = [sum(lats)/len(lats), sum(lons)/len(lons)]

    m = folium.Map(location=center, zoom_start=18, tiles="OpenStreetMap")

    tl_lon, tl_lat = tr.transform(*top_left_corner)
    br_lon, br_lat = tr.transform(*bottom_right_corner)
    s,n = min(tl_lat,br_lat), max(tl_lat,br_lat)
    w,e = min(tl_lon,br_lon), max(tl_lon,br_lon)

    sat = folium.FeatureGroup(name="Satellite Image")
    folium.raster_layers.ImageOverlay(image=satellite_image,
        bounds=[[s,w],[n,e]], opacity=0.85,
        interactive=False, cross_origin=False).add_to(sat)
    sat.add_to(m)

    has_cea_names = any(
        "cea_name" in f.get("properties", {}) for f in osm_geojson.get("features", [])
    )
    osm = folium.FeatureGroup(name="Zone Buildings" if has_cea_names else "OSM Buildings")
    tooltip_fields = ["cea_name", "building"] if has_cea_names else ["osm_id", "building"]
    tooltip_aliases = ["CEA ID", "Type"] if has_cea_names else ["OSM ID", "Type"]
    folium.GeoJson(
        osm_geojson,
        style_function=lambda _: {"fillColor":"#3388ff","color":"#1a55cc",
                                   "weight":2,"fillOpacity":0.3},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields,
                                       aliases=tooltip_aliases, localize=True)
    ).add_to(osm)
    osm.add_to(m)

    pred = folium.FeatureGroup(name="Model Predictions")
    for i, building in enumerate(predicted_buildings):
        if not building.box_coords_in_epsg_3763: continue
        n_planes = len(building.planes_in_physical_dimensions)
        folium.Polygon(locations=box_poly(building.box_coords_in_epsg_3763),
            color="#cc0000", fill_color="#ff4444", fill_opacity=0.3, weight=2,
            tooltip=f"Building {i} â€” {n_planes} roof plane(s)").add_to(pred)
        for plane in building.planes_in_physical_dimensions:
            cx = sum(p[0] for p in plane.corners)/4
            cy = sum(p[1] for p in plane.corners)/4
            lon, lat = tr.transform(cx, cy)
            folium.CircleMarker(location=[lat,lon], radius=4,
                color="#ff9900", fill=True, fill_opacity=0.8,
                tooltip=(f"Face: {plane.face}<br>Prob: {plane.probability:.2f}<br>"
                         f"Inc: {plane.inclination:.2f}<br>Ori: {plane.orientation:.2f}")
            ).add_to(pred)
    pred.add_to(m)

    # Minimum bounding boxes (rotated rectangles from cv2.minAreaRect)
    if matched_data:
        mbb = folium.FeatureGroup(name="Min Bounding Boxes")
        for entry in matched_data:
            oriented_rect = entry.get("roof_planes")
            if oriented_rect is None:
                continue
            # cv2.boxPoints returns the 4 corners of the rotated rectangle in EPSG:3763
            corners_3763 = cv2.boxPoints(oriented_rect).astype(float)  # shape (4, 2)
            # Close the ring by appending the first point again
            corners_latlon = []
            for x, y in corners_3763:
                lon, lat = tr.transform(x, y)
                corners_latlon.append([lat, lon])
            corners_latlon.append(corners_latlon[0])  # close polygon
            building_id = _entry_building_id(entry)
            _, (w_rect, h_rect), angle = oriented_rect
            folium.Polygon(
                locations=corners_latlon,
                color="#00cc66", fill_color="#00ff88", fill_opacity=0.25, weight=2,
                dash_array="6",
                tooltip=(
                    f"Building {building_id} - Min Bounding Box<br>"
                    f"W: {w_rect:.1f} m  H: {h_rect:.1f} m<br>"
                    f"Angle: {angle:.1f} deg<br>"
                    f"Code: {entry.get('code')}"
                ),
            ).add_to(mbb)
        mbb.add_to(m)


    # Roof topology lines (rotated into world coordinates)
    if matched_data:
        topo_layer = folium.FeatureGroup(name="Roof Topology Lines")
        n_lines_total = 0
        for entry in matched_data:
            lines_world = entry.get("lines_world", [])
            building_id = _entry_building_id(entry)
            print(f"Building {building_id}: {len(lines_world)} topology line(s)")
            for line in lines_world:
                p0, p1 = line
                lon0, lat0 = tr.transform(p0[0], p0[1])
                lon1, lat1 = tr.transform(p1[0], p1[1])
                print(f"  line latlon: ({lat0:.6f},{lon0:.6f}) â†’ ({lat1:.6f},{lon1:.6f})")
                folium.PolyLine(
                    locations=[[lat0, lon0], [lat1, lon1]],
                    color="#ffffff", weight=2, opacity=0.9,
                    tooltip=f"Building {building_id} - roof line",
                ).add_to(topo_layer)
                n_lines_total += 1
        print(f"Total topology lines drawn: {n_lines_total}")
        topo_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()


def _normalise_polygon_ring(polygon_ring_lon_lat):
    if len(polygon_ring_lon_lat) < 3:
        raise ValueError("Polygon ring must contain at least three points.")

    ring = []
    for point in polygon_ring_lon_lat:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Each polygon point must contain lon and lat values.")
        ring.append([float(point[0]), float(point[1])])

    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _extract_ring_from_geojson_obj(payload):
    geometry = payload
    payload_type = str(payload.get("type", "")).strip()

    if payload_type == "FeatureCollection":
        features = payload.get("features", [])
        if not isinstance(features, list) or not features:
            raise ValueError("GeoJSON FeatureCollection has no features.")
        feature = features[0]
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
    elif payload_type == "Feature":
        geometry = payload.get("geometry")

    if not isinstance(geometry, dict):
        raise ValueError("GeoJSON geometry is missing.")

    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON Polygon has no coordinates.")
        return coordinates[0]
    if geometry_type == "MultiPolygon":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON MultiPolygon has no coordinates.")
        first_polygon = coordinates[0]
        if not isinstance(first_polygon, list) or not first_polygon:
            raise ValueError("GeoJSON MultiPolygon first polygon is invalid.")
        return first_polygon[0]
    if geometry_type == "LineString":
        if not isinstance(coordinates, list) or not coordinates:
            raise ValueError("GeoJSON LineString has no coordinates.")
        return coordinates
    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def parse_polygon_text_to_ring(polygon_text):
    text = str(polygon_text).strip()
    if not text:
        raise ValueError("Polygon text is empty.")

    payload = json.loads(text)
    if isinstance(payload, list):
        ring = payload
    elif isinstance(payload, dict):
        ring = _extract_ring_from_geojson_obj(payload)
    else:
        raise ValueError("Polygon text must be either a JSON list or a GeoJSON object.")

    return _normalise_polygon_ring(ring)


def _read_text_file(path):
    with open(path, "r", encoding="utf-8") as fp:
        return fp.read()


def run_from_geojson(
    polygon_geojson,
    building_threshold=0.15,
    overlap_threshold=0.25,
    zone_shp_path=None,
    output_path="roof_surfaces.geojson",
    write_map_output=False,
):
    if isinstance(polygon_geojson, dict):
        polygon_text = json.dumps(polygon_geojson, ensure_ascii=True)
    else:
        polygon_text = str(polygon_geojson)
    ring = parse_polygon_text_to_ring(polygon_text)
    return run_from_polygon_ring(
        polygon_ring_lon_lat=ring,
        building_threshold=building_threshold,
        overlap_threshold=overlap_threshold,
        zone_shp_path=zone_shp_path,
        output_path=output_path,
        write_map_output=write_map_output,
    )


def run_from_polygon_ring(
    polygon_ring_lon_lat,
    building_threshold=0.15,
    overlap_threshold=0.25,
    zone_shp_path=None,
    output_path="roof_surfaces.geojson",
    write_map_output=False,
):
    ring = _normalise_polygon_ring(polygon_ring_lon_lat)
    lons = [lon for lon, _ in ring[:-1]]
    lats = [lat for _, lat in ring[:-1]]
    if not lons or not lats:
        raise ValueError("Polygon ring has no valid points.")

    bbox = {
        "north": max(lats),
        "south": min(lats),
        "east": max(lons),
        "west": min(lons),
    }
    jsonbox = json.dumps(bbox, ensure_ascii=True, separators=(",", ":"))
    return use_this_function(
        jsonbox,
        building_threshold,
        overlap_threshold,
        zone_shp_path=zone_shp_path,
        output_path=output_path,
        write_map_output=write_map_output,
        polygon_ring_lon_lat=ring,
    )


def use_this_function(
    jsonbox,
    building_threshold,
    overlap_threshold,
    zone_shp_path=None,
    output_path="roof_surfaces.geojson",
    write_map_output=True,
    polygon_ring_lon_lat=None,
):
    bbox = json.loads(jsonbox)
    north = float(bbox["north"])
    south = float(bbox["south"])
    east = float(bbox["east"])
    west = float(bbox["west"])

    if north <= south:
        raise ValueError("BBox north must be greater than south.")
    if east <= west:
        raise ValueError("BBox east must be greater than west.")

    if polygon_ring_lon_lat is None:
        polygon_ring_lon_lat = [
            [west, north],
            [east, north],
            [east, south],
            [west, south],
            [west, north],
        ]
    polygon_ring_lon_lat = _normalise_polygon_ring(polygon_ring_lon_lat)

    output_path = os.path.abspath(output_path)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    print("Converting coordinates...")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    tl_x, tl_y = t.transform(west, north)
    br_x, br_y = t.transform(east, south)
    area_m2 = abs(br_x - tl_x) * abs(tl_y - br_y)
    print(f"Area: {area_m2 / 1e6:.4f} km2")
    print("EPSG:4326 -> EPSG:3763 done")

    print("Fetching scenario zone.shp buildings...")
    geojson = build_cea_ordered_buildings_geojson(
        polygon_ring_lon_lat, zone_shp_path=zone_shp_path
    )
    print(f"Zone done: {len(geojson['features'])} footprint(s)")

    print("Connecting to WMTS...")
    wmts_url = (
        "https://cartografia.dgterritorio.gov.pt/ortos2018/service"
        "?service=WMTS&request=GetCapabilities"
    )
    wmts = WebMapTileService(wmts_url)
    matrix = wmts.tilematrixsets["PTTM_06"].tilematrix["14"]
    col_min, col_max, row_min, row_max = get_tile_indices(tl_x, br_y, br_x, tl_y, matrix)
    total_tiles = (col_max + 1 - col_min) * (row_max + 1 - row_min)
    print(f"Service ready: {total_tiles} tile(s) to download")

    def tile_cb(done, total):
        print(f"Downloading tiles... {done}/{total}")

    satellite_image, res, _ = retrieve_satelite_image((tl_x, tl_y), (br_x, br_y), progress_cb=tile_cb)
    h, w = satellite_image.shape[:2]

    print("Running YOLO inference...")
    print(f"Running detector (conf>={building_threshold:.2f}, iou<={overlap_threshold:.2f})...")
    img_bgr = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)
    buildings = retrieve_prediction_list(
        img_bgr,
        (tl_x, tl_y),
        res,
        building_threshold,
        overlap_threshold,
        model,
    )
    print(f"Detected {len(buildings)} building(s)")

    annotated_image = draw_azimuths_on_satellite(satellite_image, buildings, (tl_x, tl_y), res)
    if write_map_output:
        annotated_path = os.path.join(output_parent or SCRIPT_DIR, "satellite_image.png")
        Image.fromarray(annotated_image).save(annotated_path)
        print(f"Satellite image saved to: {annotated_path} ({w}x{h} px, {res:.3f} m/px)")
    else:
        print(f"Satellite image ready ({w}x{h} px, {res:.3f} m/px)")

    matched_data = []
    if buildings:
        print(f"Computing IOU matrix of size [{len(geojson['features'])},{len(buildings)}]")
        iou_mat = compute_iou_matrix(geojson["features"], buildings)
        print("Done computing IOU matrix.")
        n_features = len(geojson["features"])
        for i, osm_feat in enumerate(geojson["features"]):
            cea_name = str(osm_feat.get("properties", {}).get("cea_name", "")).strip()
            if not cea_name or cea_name.lower() == "nan":
                print(
                    f"[warn] Feature index {i} has no valid CEA name. Skipping to preserve ID integrity."
                )
                continue

            best_match_idx = int(np.argmax(iou_mat[i, :]))
            best_iou = float(iou_mat[i, best_match_idx])
            print(f"[match] Building {cea_name}: best_pred={best_match_idx}, IoU={best_iou:.4f}")
            if best_iou > 0.3:
                pred = buildings[best_match_idx]
                outline, orientedbox, lines_world, code, face_data = topology_converter_mine(
                    pred.raw_roof_data, osm_feat
                )
                matched_data.append(
                    {
                        "building_id": cea_name,
                        "cea_name": cea_name,
                        "footprint": outline,
                        "roof_planes": orientedbox,
                        "lines_world": lines_world,
                        "code": code,
                        "face_data": face_data,
                    }
                )
            if i % 10 == 0 or i == n_features - 1:
                print(f"Matching {i + 1}/{n_features} - {len(matched_data)} matched so far")
    else:
        print("[warn] No model predictions found. Skipping matching.")

    export_roof_surfaces_geojson(matched_data, output_path=output_path)
    print(f"[ok] Roof surfaces exported to {output_path}")

    if write_map_output:
        print("Rendering map...")
        map_html = build_map_html(
            geojson,
            buildings,
            satellite_image,
            (tl_x, tl_y),
            (br_x, br_y),
            matched_data=matched_data,
        )
        map_path = os.path.join(output_parent or SCRIPT_DIR, "building_map.html")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(map_html)
        print(f"Map saved successfully to: {map_path}")

        print("Rendering pydeck map...")
        render_pydeck(geojson, matched_data)
        print("All done.")

    return {
        "output_path": output_path,
        "matched_buildings": len(matched_data),
        "zone_features": len(geojson["features"]),
    }


def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description="Generate roof_surfaces.geojson from tiled fixed-box roof predictions."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--polygon-ring",
        help=(
            "Polygon coordinates as JSON ring [[lon,lat],...] or GeoJSON text "
            "(FeatureCollection, Feature, Polygon, MultiPolygon, or LineString)."
        ),
    )
    source_group.add_argument(
        "--polygon-geojson",
        help="GeoJSON string (for example a FeatureCollection) describing the polygon area.",
    )
    source_group.add_argument(
        "--polygon-geojson-file",
        help="Path to a .geojson/.json file containing the polygon (FeatureCollection/Feature/Polygon).",
    )
    parser.add_argument(
        "--zone-shp-path",
        default=DEFAULT_ZONE_SHP_PATH,
        help="Path to scenario zone.shp used for authoritative CEA building ids.",
    )
    parser.add_argument(
        "--output-path",
        default="roof_surfaces.geojson",
        help="Path to write the roof surfaces GeoJSON output.",
    )
    parser.add_argument(
        "--building-threshold",
        type=float,
        default=0.15,
        help="Detection confidence threshold for buildings.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.25,
        help="NMS overlap threshold for deduplicating predictions.",
    )
    parser.add_argument(
        "--write-map-output",
        action="store_true",
        help="Also export satellite and map visualisation files next to output-path.",
    )
    return parser


def main():
    parser = _build_cli_parser()
    args = parser.parse_args()
    polygon_text = args.polygon_ring
    if args.polygon_geojson:
        polygon_text = args.polygon_geojson
    elif args.polygon_geojson_file:
        polygon_text = _read_text_file(args.polygon_geojson_file)

    polygon_ring_lon_lat = parse_polygon_text_to_ring(polygon_text)
    result = run_from_polygon_ring(
        polygon_ring_lon_lat=polygon_ring_lon_lat,
        building_threshold=args.building_threshold,
        overlap_threshold=args.overlap_threshold,
        zone_shp_path=args.zone_shp_path,
        output_path=args.output_path,
        write_map_output=args.write_map_output,
    )
    print(
        "[ok] Finished fixedboxtilingextension "
        f"(zone_features={result['zone_features']}, matched_buildings={result['matched_buildings']})"
    )


if __name__ == "__main__":
    main()
