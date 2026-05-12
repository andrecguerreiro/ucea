import numpy as np
from ultralytics import YOLO
from owslib.wmts import WebMapTileService
from PIL import Image
import io
import cv2
import folium
import math
from pyproj import Transformer
import requests
import json
import gradio as gr
import os
import hashlib
import argparse
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

## Listing of logic in the project
'''
We download high resolution satellite images. 
From this images we retrieve predictions with our model.
Then we retrieve the OSM buildings from openstreetmap.
Then we match these buildings.
We recreate a rooftop based on the building outline and the model prediction 
We render it
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

    os.makedirs("cache/tiles", exist_ok=True)

    done = 0
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tile_filename = f"cache/tiles/tile_{zoom_level}_{row}_{col}.png"
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
        self.model = YOLO("best.pt")
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

        inc_top = 0.308 
        inc_right = 0.308 
        inc_bottom = 0.308 
        inc_left = 0.308 

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
    q = f"""[out:json];(way["building"]({s},{w},{n},{e});relation["building"]({s},{w},{n},{e}););out body;>;out skel qt;"""
    #r = requests.post("https://overpass-api.de/api/interpreter", data=q)
    ## AI suggested solution
    headers = {
        'User-Agent': 'OVEN_Building_Tool/4.0 (https://github.com/Joaopmoliveira/OVEN)',
    }
    payload = {'data': q}

    r = requests.post(
        "https://overpass-api.de/api/interpreter", 
        data=payload, 
        headers=headers
    )
    #r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q})
    #r = requests.post(
    #    "https://overpass-api.de/api/interpreter",
    #    data={"data": q},          # <-- the fix
    #    timeout=60                 # good practice for external APIs
    #)
    print("Query sent:\n", q)
    print("Status:", r.status_code)
    print("Response:", r.text[:500])  # Overpass usually returns an error message
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
    os.makedirs("cache/osm", exist_ok=True)
    
    # Create a unique ID based on the coordinates
    coord_str = f"{top_left}_{bottom_right}"
    cache_key = hashlib.md5(coord_str.encode()).hexdigest()
    cache_path = f"cache/osm/{cache_key}.json"

    if os.path.exists(cache_path):
        print(f"ðŸ“¦ Loading OSM data from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    # If not cached, fetch it
    data = get_osm_buildings(top_left, bottom_right)
    
    with open(cache_path, "w") as f:
        json.dump(data, f)
    return data

import numpy as np
import math
from pyproj import Transformer
from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, GeometryCollection
from shapely.ops import split, polygonize, unary_union
import cv2

def convert_azimuth_to_local_vector_in_bounding_box(face_id,ori,axis_aligned_bounding_box_rotation):
    if face_id == "T":
        base_ori_rad = math.pi / 2
    elif face_id == "B":
        base_ori_rad = -math.pi / 2 
    elif face_id == "R":
        base_ori_rad = 0 
    elif face_id == "L":
        base_ori_rad = math.pi 
    elif face_id == "H":
        return [0, 1],0.0
    else:
        return None,0.0
    azimuth = (ori - 0.5) * (math.pi / 2) + base_ori_rad - axis_aligned_bounding_box_rotation
    nx = math.cos(azimuth)
    ny = math.sin(azimuth)
    return [nx, ny]   

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
    Returns the face identifier (e.g., 'T', 'R', 'B', 'L').
    Never returns 'H' directly â€” H assignment is handled by the caller via
    a remainder strategy (match all directional faces first, leftover â†’ H).

    corners: [[x_min_l, y_min_l], [x_max_l, y_min_l], [x_max_l, y_max_l], [x_min_l, y_max_l]]
    """
    x_min_l, y_min_l = corners[0]
    x_max_l, y_max_l = corners[2]
    cx, cy = poly_centroid

    # Distance from centroid to each directional edge.
    # H is intentionally excluded â€” it is never matched by proximity.
    distances = {
        'T': abs(cy - y_max_l),
        'B': abs(cy - y_min_l),
        'R': abs(cx - x_max_l),
        'L': abs(cx - x_min_l),
    }

    # Only consider directional faces that are present in this code
    valid_distances = {face: dist for face, dist in distances.items() if face in code}

    if valid_distances:
        return min(valid_distances, key=valid_distances.get)

    raise NameError('There must be one valid distance') 

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import triangulate

def to_triangulated_pyvista(pts_2d):
    """Convert a potentially concave 2D polygon to a triangulated PyVista mesh."""
    import pyvista as pv
    poly = ShapelyPolygon(pts_2d)
    triangles = triangulate(poly)  # Delaunay triangulation
    
    all_pts = []
    faces = []
    offset = 0
    for tri in triangles:
        if not poly.contains(tri.centroid):
            continue  # discard triangles outside the polygon (concavity artifacts)
        coords = np.array(tri.exterior.coords[:-1])
        coords_3d = np.column_stack([coords, np.zeros(3)])
        all_pts.append(coords_3d)
        faces.extend([3, offset, offset+1, offset+2])
        offset += 3
    
    all_pts = np.vstack(all_pts)
    return pv.PolyData(all_pts, faces=faces).extrude((0, 0, 1000), capping=True)

def rooftile(corners, plane_point, plane_data, height,osm_local_2d):
    """
    This function computes the intersection of the corners of 
    the section of the building reserved for this direction
    with the roof predicted by the OVEN model
    """
    import pyvista as pv
    from shapely.geometry import Polygon, MultiPolygon
    plane_normal,azymuth = plane_data 

    rect_poly = ShapelyPolygon(corners if np.array(corners).shape[1] == 2 else np.array(corners)[:, :2])
    osm_poly  = ShapelyPolygon(osm_local_2d)
    clipped = rect_poly.intersection(osm_poly)

    if clipped.is_empty:
        return None, azymuth

    if isinstance(clipped, MultiPolygon):
        clipped = max(clipped.geoms, key=lambda p: p.area)

    pts_2d = np.array(clipped.exterior.coords[:-1])  # shape (N, 2)
    pts = np.column_stack([pts_2d, np.zeros(len(pts_2d))])  # add Z=0
    n_pts = len(pts)
    faces = [n_pts] + list(range(n_pts))

    ## Ok if I am right, what we need to do is to do is:
    # 1 . we have a rect_prism that is the outline section of the rooftop which has infite high
    # 2 . we have the potentially cropped building outline 
    # 3 . we fuse them both
    # 4 . now when we cut the building with the plane we return this extra cutted 3D shape
    # 5 . we then return the 
    rect_prism = pv.PolyData(pts, faces=faces).extrude((0, 0, height), capping=True).triangulate() 
    intersection_polygon = rect_prism.slice(normal=plane_normal, origin=plane_point)
    return intersection_polygon,azymuth
 
def match_vector_to_coordinates(dx, dy, width, height):
    """
    Returns the index of the first side hit by the ray (dx, dy) from origin.
    Side indexing matches your comment:
      0 -> side1: top    (c11-c01, y = +height/2)
      1 -> side2: bottom (c00-c10, y = -height/2)
      2 -> side3: left   (c00-c01, x = -width/2)
      3 -> side4: right  (c11-c10, x = +width/2)
    """
    half_w = width  / 2.0
    half_h = height / 2.0

    candidates = []  # (t, side_index)

    # Horizontal sides (solved via y component)
    if dy != 0:
        t_top    = half_h  / dy   # side1: y = +half_h
        t_bottom = -half_h / dy   # side2: y = -half_h
        if t_top > 0:
            x_hit = dx * t_top
            if -half_w <= x_hit <= half_w:
                candidates.append((t_top, 0))   # top
        if t_bottom > 0:
            x_hit = dx * t_bottom
            if -half_w <= x_hit <= half_w:
                candidates.append((t_bottom, 1))  # bottom

    # Vertical sides (solved via x component)
    if dx != 0:
        t_right = half_w  / dx   # side4: x = +half_w
        t_left  = -half_w / dx   # side3: x = -half_w
        if t_right > 0:
            y_hit = dy * t_right
            if -half_h <= y_hit <= half_h:
                candidates.append((t_right, 3))  # right
        if t_left > 0:
            y_hit = dy * t_left
            if -half_h <= y_hit <= half_h:
                candidates.append((t_left, 2))   # left

    if not candidates:
        return None  # Ray is parallel to both axes (zero vector)

    # The first hit is the smallest positive t
    t_hit, side_idx = min(candidates, key=lambda c: c[0])
    side_names = ["T", "B", "L", "R"]
    return side_names[side_idx]

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
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    pts_world = outline[:2, :].T  # shape (N, 2)
    pts_centered = pts_world - np.array(center)
    osm_local = np.column_stack([
        pts_centered[:, 0] * cos_t + pts_centered[:, 1] * sin_t,
        -pts_centered[:, 0] * sin_t + pts_centered[:, 1] * cos_t,
    ])

    x_min_l = -width / 2.0
    x_max_l = width / 2.0
    y_min_l = -height / 2.0
    y_max_l = height / 2.0
    corners = np.array([
        [x_min_l, y_min_l],
        [x_max_l, y_min_l],
        [x_max_l, y_max_l],
        [x_min_l, y_max_l],
    ])
    import pyvista as pv
    from shapely.geometry import Polygon, MultiPolygon
    pts = np.array(corners)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts))])
    n_pts = len(pts)
    faces = [n_pts] + list(range(n_pts))
    building_cuboid = pv.PolyData(pts, faces=faces).extrude((0, 0, height), capping=True)

    #    corners = np.array([
    #    c00,
    #    c10,
    #    c11,
    #    c01
    #]) 
    # :top  is c11 - c01
    # :bottom  is c00 - c10
    # :left  is c00 - c01
    # :right  is c11 - c10

    # after rotating all the corners, each one will be a top, bottom, etc.. 
 
    theta = math.radians(angle)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
 
    global_probs = {"T": probtop, "R": probright, "B": probbottom, "L": probleft}
    global_inc = {"T": 0.308, "R": 0.308, "B": 0.308, "L": 0.308}
    global_ori = {"T": ori_top, "R": ori_right, "B": ori_bottom, "L": ori_left}

    face_data = {
        "T": {"active": global_probs["T"] > 0.5, "inclination": global_inc["T"], "orientation": global_ori["T"]},
        "R": {"active": global_probs["R"] > 0.5, "inclination": global_inc["R"], "orientation": global_ori["R"]},
        "B": {"active": global_probs["B"] > 0.5, "inclination": global_inc["B"], "orientation": global_ori["B"]},
        "L": {"active": global_probs["L"] > 0.5, "inclination": global_inc["L"], "orientation": global_ori["L"]},
        "H": {"active": probh > 0.5, "inclination": 0.0, "orientation": 0.0},
    }
 
    codelocalcoords = []
    ## Ok important, we need to deal with the case where the probability of none is higher than 50% 
    # but one must be there because a building was identified. 
    # What I suggest is the following: We select the highest probability, and check minus 0.05 percent bellow that maximum
    maximumprob = roof_prediction[0:5].max()

    maximumprob =  0.5*0.8 if maximumprob > 0.5 else maximumprob*0.80   
    if probh > maximumprob: 
        codelocalcoords.append(["H","H",probh,face_data["H"]])
        #code += "H"
    if global_probs["T"] > maximumprob: 
        vector = convert_azimuth_to_local_vector_in_bounding_box("T",face_data["T"]["orientation"],theta)
        dx,dy = vector
        transformed_face = match_vector_to_coordinates(dx, dy, width, height)
        codelocalcoords.append([transformed_face,"T",global_probs["T"],face_data["T"]])
        #code += "T"
    if global_probs["R"] > maximumprob: 
        vector = convert_azimuth_to_local_vector_in_bounding_box("R",face_data["R"]["orientation"],theta)
        dx,dy = vector
        transformed_face = match_vector_to_coordinates(dx, dy, width, height)
        codelocalcoords.append([transformed_face,"R",global_probs["R"],face_data["R"]])
        #code += "R"
    if global_probs["B"] > maximumprob: 
        vector = convert_azimuth_to_local_vector_in_bounding_box("B",face_data["B"]["orientation"],theta)
        dx,dy = vector
        transformed_face = match_vector_to_coordinates(dx, dy, width, height)
        codelocalcoords.append([transformed_face,"B",global_probs["B"],face_data["B"]])
        #code += "B"
    if global_probs["L"] > maximumprob: 
        vector = convert_azimuth_to_local_vector_in_bounding_box("L",face_data["L"]["orientation"],theta)
        dx,dy = vector
        transformed_face = match_vector_to_coordinates(dx, dy, width, height)
        codelocalcoords.append([transformed_face,"L",global_probs["L"],face_data["L"]])
        #code += "L" 

    parsed = {}
    for localface,globalface,probability,data in codelocalcoords:
        possible_old_value = parsed.get(localface)
        if possible_old_value is None:
            parsed[localface] = [globalface,probability,data]
        else:
            old_global_face,old_probability,old_data = possible_old_value
            if old_probability < probability:
                parsed[localface] = [globalface,probability,data]

    code = ""
    for key,val in parsed.items():
        code = code + key
    code = "".join(sorted(code)) ## we need to sort or else HRL and HLR would be different, which is not the case

    if len(code) == 0: #this is a sanity check
        raise NameError('The code must never be empty. If a building exists, then at least one rooftop is present')

    lines = []
    
    intersections = {}
    
    base_height = 10.0
    
    def hungarian_internal_function(matched_code, parsed, data, theta, corners, intersections, base_height, parts, osm_local,intersection_point):
        from scipy.optimize import linear_sum_assignment
        assert len(parts) == len(matched_code), "division of rooftop encountered an error, please check geometry"+matched_code

        x_min_l, y_min_l = corners[0]
        x_max_l, y_max_l = corners[2]

        def _edge_dist(face, cx, cy):
            if face == 'T': return abs(cy - y_max_l)
            if face == 'B': return abs(cy - y_min_l)
            if face == 'R': return abs(cx - x_max_l)
            if face == 'L': return abs(cx - x_min_l)
            return float('inf')

        centroids = [np.mean(part, axis=0) for part in parts]
        non_h_faces = [f for f in matched_code if f != 'H']

        facedict = {}
        for letter in matched_code:
            global_face, probability, fdata = parsed[letter]
            facedict[letter] = planenormal(letter, fdata["inclination"], fdata["orientation"], theta)   


        if non_h_faces:
            # Cost matrix: only directional faces vs all parts
            cost = np.array([
                [_edge_dist(face, cx, cy) for cx, cy in centroids]
                for face in non_h_faces
            ])
            face_indices, part_indices = linear_sum_assignment(cost)
            assigned_parts = set(part_indices)

            for fi, pi in zip(face_indices, part_indices):
                face = non_h_faces[fi]
                intersections[face] = rooftile(parts[pi], intersection_point, facedict[face], 1000, osm_local)
        else:
            assigned_parts = set()

        # H gets the leftover part â€” no assumption about its position
        if 'H' in matched_code:
            leftover = [i for i in range(len(parts)) if i not in assigned_parts]
            assert len(leftover) == 1, f"Expected exactly 1 leftover part for H, got {len(leftover)}"
            intersections['H'] = rooftile(parts[leftover[0]], intersection_point, facedict['H'], 1000, osm_local)

    ''' TODO this is the old version of the internal function that I am keeping for consistency, TO ELIMINATE
    def old_internal_function(matched_code,parsed,data,theta,corners,intersections,base_height,parts,osm_local):
        assert len(parts) == len(matched_code), "division of rooftop encountered an error, please check geometry"
        facedict = {}
        for letter in matched_code:
            global_face,probability,data = parsed[letter]
            normal = planenormal(letter, data["inclination"], data["orientation"],theta)
            facedict[letter] = normal
        if 'H' not in matched_code: # ok in this case the assignment logic works fine
            for part in parts:
                centroid = np.mean(part, axis=0)
                face = determine_face_for_polygon(centroid, corners, matched_code)
                normal = facedict.get(face)
                if normal is not None:
                    intersections[face] = rooftile(part, np.array([0, 0, base_height]), normal,1000,osm_local)
                else:
                    print("failure detected")
        else : # this is the hard case. here we need to match all other faces and only then select the remaining part to be the horizontal tile
            non_h_faces = [f for f in matched_code if f != 'H']
            centroids = [np.mean(part, axis=0) for part in parts]
            x_min_l, y_min_l = corners[0]
            x_max_l, y_max_l = corners[2]

            def _edge_dist(face, cx, cy):
                if face == 'T': return abs(cy - y_max_l)
                if face == 'B': return abs(cy - y_min_l)
                if face == 'R': return abs(cx - x_max_l)
                if face == 'L': return abs(cx - x_min_l)
                return float('inf')
            unassigned = list(range(len(parts)))
            face_assignments = {}  # face -> part index

            # Greedy: for each non-horizontal tile, claim the closest still-unassigned part.
            for face in non_h_faces:
                if not unassigned:
                    break
                best_idx = min(unassigned,
                               key=lambda i: _edge_dist(face, *centroids[i]))
                face_assignments[face] = best_idx
                unassigned.remove(best_idx)
            if len(unassigned) != 1:
                raise NameError('The the unassigned should be one')

            face_assignments['H'] = unassigned[0]

            for face, idx in face_assignments.items():
                normal = facedict.get(face)
                if normal is not None:
                    intersections[face] = rooftile(parts[idx], np.array([0, 0, base_height]), normal,1000,osm_local)
                else:
                    raise Exception("The face should be assigned to something")
    '''
    match code:
        case "L":
            global_face,probability,data = parsed["L"]
            normal = planenormal("L", data["inclination"], data["orientation"],theta)
            intersections["L"] = rooftile(corners, np.array([0, 0, base_height]), normal,1000,osm_local)
            
        case "B":
            global_face,probability,data = parsed["B"]
            normal = planenormal("B",data["inclination"], data["orientation"],theta)
            intersections["B"] = rooftile(corners, np.array([0, 0, base_height]), normal,1000,osm_local)
            
        case "R":
            global_face,probability,data = parsed["R"]
            normal = planenormal("R", data["inclination"], data["orientation"],theta)
            intersections["R"] = rooftile(corners, np.array([0, 0, base_height]), normal,1000,osm_local)
            
        case "T":
            global_face,probability,data = parsed["T"]
            normal = planenormal("T", data["inclination"], data["orientation"],theta)
            intersections["T"] = rooftile(corners, np.array([0, 0, base_height]), normal,1000,osm_local)
            
        case "H":
            global_face,probability,data = parsed["H"]
            normal = planenormal("H", data["inclination"], data["orientation"],theta)
            intersections["H"] = rooftile(corners, np.array([0, 0, base_height]), normal,1000,osm_local)
            
        case "BL": #corrected
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
        case "LR": #corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))                    
        case "BR": #corrected
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HL":#corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BH":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "LT":#corrected
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BT":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "RT":#corrected
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HR":#corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HT":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BLR":#corrected
            lines.append([[0, y_max_l], [x_min_l, y_min_l]])
            lines.append([[0, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, y_max_l, base_height]))
                    
        case "BLT":#corrected
            lines.append([[x_max_l, 0], [x_min_l, y_min_l]])
            lines.append([[x_max_l, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([x_max_l, 0, base_height]))
                    
        case "LRT":#corrected
            lines.append([[0, y_min_l], [x_min_l, y_max_l]])
            lines.append([[0, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, y_min_l, base_height]))
                    
        case "BRT":#corrected
            lines.append([[x_min_l, 0], [x_max_l, y_min_l]])
            lines.append([[x_min_l, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([x_min_l, 0, base_height]))
                    
        case "HLT":#corrected
            lines.append([[0, 0], [x_min_l, y_max_l]])
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHT":#corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            lines.append([[0, 0], [x_max_l, 0]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHL":#corrected
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_min_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HLR":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            lines.append([[0, 0], [0, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHR":#corrected
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HRT":#corrected
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BLRT":#corrected
            lines.append([[x_min_l, y_min_l], [x_max_l, y_max_l]])
            lines.append([[x_min_l, y_max_l], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHLR":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            parts = split_with_lines(corners, lines)
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHLT":#corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            lines.append([[0, 0], [x_min_l, y_min_l]])
            lines.append([[0, 0], [x_min_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "HLRT":#corrected
            lines.append([[x_min_l, 0], [x_max_l, 0]])
            lines.append([[0, 0], [x_min_l, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHRT":#corrected
            lines.append([[0, y_min_l], [0, y_max_l]])
            lines.append([[0, 0], [x_max_l, y_min_l]])
            lines.append([[0, 0], [x_max_l, y_max_l]])
            parts = split_with_lines(corners, lines)
            
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
                    
        case "BHLRT":
            ix_min, ix_max = 0.5 * x_min_l, 0.5 * x_max_l
            iy_min, iy_max = 0.5 * y_min_l, 0.5 * y_max_l
            lines.append([[ix_min, iy_min], [ix_max, iy_min]])
            lines.append([[ix_max, iy_min], [ix_max, iy_max]])
            lines.append([[ix_max, iy_max], [ix_min, iy_max]])
            lines.append([[ix_min, iy_max], [ix_min, iy_min]])
            parts = [
                np.array([[ix_min, iy_min], [ix_max, iy_min],
                           [ix_max, iy_max], [ix_min, iy_max]]),
                np.array([[x_min_l, y_max_l], [x_max_l, y_max_l],
                           [ix_max,  iy_max], [ix_min,  iy_max]]),
                np.array([[x_min_l, y_min_l], [ix_min,  iy_min],
                           [ix_max,  iy_min], [x_max_l, y_min_l]]),
                np.array([[x_min_l, y_min_l], [x_min_l, y_max_l],
                           [ix_min,  iy_max], [ix_min,  iy_min]]),
                np.array([[x_max_l, y_min_l], [ix_max,  iy_min],
                           [ix_max,  iy_max], [x_max_l, y_max_l]]),
            ]
            hungarian_internal_function(code,parsed,data,theta,corners,intersections,base_height,parts,osm_local, np.array([0, 0, base_height]))
    
    # Convert lines to world coordinates
    cx, cy = center
    for face in intersections:
        # Double the count of every intersection
        intersection,azymuth = intersections[face]
        if intersection is not None:
            intersection.rotate_z(angle, inplace=True)
            intersection.translate([cx, cy, 0], inplace=True)

    selected_faces = {}
    for local_face, values in parsed.items():
        global_face, probability, data = values
        selected_faces[str(local_face)] = {
            "global_face": str(global_face) if global_face is not None else None,
            "probability": float(probability) if probability is not None else None,
            "inclination": data.get("inclination") if isinstance(data, dict) else None,
            "orientation": data.get("orientation") if isinstance(data, dict) else None,
        }

    # Store selections and intersections in face_data for downstream export.
    face_data["selected_faces"] = selected_faces
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

import pydeck as pdk
from pyproj import Transformer

# Ok to convert quickly we allocate one transformer that is repeatedly used
_tr_3763_to_4326 = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True)
 
def _to_lonlat(x, y):
    lon, lat = _tr_3763_to_4326.transform(x, y)
    return float(lon), float(lat)
 
def _make_geojson_layer(geojson_fc) -> pdk.Layer:
    """
    OSM building footprints â†’ GeoJsonLayer (extruded, flat-roofed reference).
    The `geojson_fc` is the FeatureCollection returned by get_osm_buildings_cached().
    Coordinates are already in EPSG:4326 (lon/lat) as required by deck.gl.
    """
    return pdk.Layer(
        "GeoJsonLayer",
        geojson_fc,                         # pass the dict directly
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
            "osm_id":  feat["properties"].get("osm_id", "?"),
        })
 
    return pdk.Layer(
        "PolygonLayer",
        records,
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
            "osm_id":  entry.get("osm_id", "?"),
            "code":    entry.get("code", ""),
        })
 
    return pdk.Layer(
        "PolygonLayer",
        records,
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
        osm_id      = entry.get("osm_id", "?")
        code = entry.get("code", {})
        for face_name, plane in intersections.items():
            if plane is None:
                continue
            points,azymuth = plane 
            raw_pts = points.points
            polygon_3d = []
            for v in raw_pts.tolist():
                lon, lat = _to_lonlat(v[0], v[1])
                polygon_3d.append([lon, lat, float(v[2])])  # z kept as metres
 
            records.append({
                "polygon":   polygon_3d,
                "azymuth": round(math.degrees(azymuth), 1),   # centroid height for tooltip
                "face":      face_name,
                "osm_id":    osm_id,
                "code" :   code ,
                "color":     _face_colors.get(face_name, _default_color),
            })
 
    return pdk.Layer(
        "PolygonLayer",
        records,
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
        osm_id     = entry.get("osm_id", "?")
        lines_world = entry.get("lines_world", [])
        for seg in lines_world:
            p0, p1 = seg
            lon0, lat0 = _to_lonlat(p0[0], p0[1])
            lon1, lat1 = _to_lonlat(p1[0], p1[1])
            records.append({
                "path":   [[lon0, lat0], [lon1, lat1]],
                "osm_id": osm_id,
            })
 
    return pdk.Layer(
        "PathLayer",
        records,
        get_path="path",
        get_color=[255, 255, 255, 230],
        get_width=0.3,          # metres
        width_min_pixels=2,
        pickable=True,
    )
 
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
        pitch=45,
        bearing=0,
        max_zoom=22,
    )
 
  
def render_pydeck(
    geojson_fc,
    matched_data,
    out_path="3d_roofs.html",
    map_style="dark",   # "dark" | "light" | "satellite" | "road"
):
    """
    Build a pydeck Deck with four layers and write it to `out_path`.
 
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
 
    view_state = _center_view(matched_data, geojson_fc)
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider="carto",
        map_style=_map_styles.get(map_style, pdk.map_styles.ROAD),
        tooltip={
            "html": (
                "<b>OSM {osm_id}</b><br/>"
                "Face: {face} | Code: {code}<br/>"
                "Azymuth: {azymuth} Âº"
            ),
            "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"},
        },
    )
 
    deck.to_html(out_path)
    print(f"âœ…  pydeck map saved â†’ {out_path}")

    html_str = deck.to_html(as_string=True)
    return html_str

def reproject_to_geographic(
    image_rgb: np.ndarray,
    xmin: float, ymin: float, xmax: float, ymax: float,
    src_crs: str = "EPSG:3763",
    ) -> tuple[np.ndarray, float, float, float, float]:
    """
    Reproject a uint8 RGB image from `src_crs` to EPSG:4326 so that pixel
    rows/columns align with lines of constant latitude/longitude.
    Without this, the TM meridian convergence (~0.67Â° near Lisbon) causes
    ImageOverlay to appear rotated, producing a ~12 m corner offset.
    Returns: reprojected image, (west, south, east, north) in degrees.
    """
    h, w = image_rgb.shape[:2]
    src_transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
    dst_crs = "EPSG:4326"
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h,
        left=xmin, bottom=ymin, right=xmax, top=ymax,
    )
    reprojected = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=image_rgb[:, :, band],
            destination=reprojected[:, :, band],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    west  =  dst_transform.c
    north =  dst_transform.f
    east  =  west  + dst_transform.a * dst_w
    south =  north + dst_transform.e * dst_h
    return reprojected, west, south, east, north

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

    xmin_3763, ymax_3763 = top_left_corner
    xmax_3763, ymin_3763 = bottom_right_corner
    sat_geo, w, s, e, n = reproject_to_geographic(
        satellite_image, xmin_3763, ymin_3763, xmax_3763, ymax_3763
    )

    sat = folium.FeatureGroup(name="Satellite Image")
    folium.raster_layers.ImageOverlay(image=sat_geo,
        bounds=[[s,w],[n,e]], opacity=0.85,
        interactive=False, cross_origin=False).add_to(sat)
    sat.add_to(m)
    osm = folium.FeatureGroup(name="OSM Buildings")
    folium.GeoJson(osm_geojson,
        style_function=lambda _: {"fillColor":"#3388ff","color":"#1a55cc",
                                   "weight":2,"fillOpacity":0.3},
        tooltip=folium.GeoJsonTooltip(fields=["osm_id","building"],
                                       aliases=["OSM ID","Type"], localize=True)
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
            osm_id = entry.get("osm_id", "?")
            _, (w_rect, h_rect), angle = oriented_rect
            folium.Polygon(
                locations=corners_latlon,
                color="#00cc66", fill_color="#00ff88", fill_opacity=0.25, weight=2,
                dash_array="6",
                tooltip=(f"OSM {osm_id} â€” Min Bounding Box<br>"
                         f"W: {w_rect:.1f} m  H: {h_rect:.1f} m<br>"
                         f"Angle: {angle:.1f}Â°"
                         f"Azymuth:{0:.1f}"
                         f"Name: {entry.get("code")}"),
            ).add_to(mbb)
        mbb.add_to(m)


    # Roof topology lines (rotated into world coordinates)
    if matched_data:
        topo_layer = folium.FeatureGroup(name="Roof Topology Lines")
        n_lines_total = 0
        for entry in matched_data:
            lines_world = entry.get("lines_world", [])
            osm_id = entry.get("osm_id", "?")
            print(f"OSM {osm_id}: {len(lines_world)} topology line(s)")
            for line in lines_world:
                p0, p1 = line
                lon0, lat0 = tr.transform(p0[0], p0[1])
                lon1, lat1 = tr.transform(p1[0], p1[1])
                print(f"  line latlon: ({lat0:.6f},{lon0:.6f}) â†’ ({lat1:.6f},{lon1:.6f})")
                folium.PolyLine(
                    locations=[[lat0, lon0], [lat1, lon1]],
                    color="#ffffff", weight=2, opacity=0.9,
                    tooltip=f"OSM {osm_id} â€” roof line",
                ).add_to(topo_layer)
                n_lines_total += 1
        print(f"Total topology lines drawn: {n_lines_total}")
        topo_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

#jsonbox = "{\"north\":38.75655020905061,\"south\":38.75086491020125,\"east\":-9.19761657714844,\"west\":-9.204397201538088}"
#building_threshold = 0.15
#overlap_threshold = 0.25

def build_cea_ordered_buildings_geojson(polygon_ring_lon_lat, zone_shp_path):
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

    if not zone_shp_path:
        raise ValueError(
            "zone_shp_path is required. Provide --zone-shp-path pointing to scenario "
            "inputs/building-geometry/zone.shp."
        )

    try:
        import geopandas as gpd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "GeoPandas is required to read scenario zone.shp footprints."
        ) from exc

    from shapely.geometry import Polygon as ShapelyPolygon

    resolved_zone_shp_path = os.path.abspath(str(zone_shp_path).strip())
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
        if not cea_name:
            continue

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
                "properties": {
                    "cea_name": cea_name,
                    "building": building_type,
                    "osm_id": cea_name,  # compatibility with existing tooltip/layer code
                },
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

def _entry_building_id(entry, fallback="?"):
    return str(
        entry.get("building_id")
        or entry.get("cea_name")
        or entry.get("osm_id")
        or fallback
    )


def _parse_vtk_cells(flat_cells):
    """
    Parse VTK-style flat cell arrays:
    [n0, p0, p1, ..., n1, p0, p1, ...]
    """
    if flat_cells is None:
        return []
    arr = np.asarray(flat_cells).astype(int, copy=False).ravel()
    if arr.size == 0:
        return []

    cells = []
    i = 0
    total = arr.size
    while i < total:
        n = int(arr[i])
        i += 1
        if n <= 0 or i + n > total:
            break
        cell = [int(x) for x in arr[i:i + n]]
        i += n
        if len(cell) >= 2:
            cells.append(cell)
    return cells


def _best_fit_plane(points_3d):
    if points_3d.shape[0] < 3 or points_3d.shape[1] < 3:
        return None
    centroid = np.mean(points_3d, axis=0)
    centered = points_3d - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) < 3:
        return None

    axis_u = vh[0]
    normal = vh[2]
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


def _to_uv(points_xyz, frame):
    centroid, axis_u, axis_v, _ = frame
    centered = points_xyz - centroid
    u = centered @ axis_u
    v = centered @ axis_v
    return np.column_stack([u, v])


def _uv_to_xyz(u, v, frame):
    centroid, axis_u, axis_v, _ = frame
    xyz = centroid + float(u) * axis_u + float(v) * axis_v
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _extract_boundary_edges_from_faces(faces_cells):
    edge_counts = {}
    for cell in faces_cells:
        if len(cell) < 3:
            continue
        for i in range(len(cell)):
            a = int(cell[i])
            b = int(cell[(i + 1) % len(cell)])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return [edge for edge, count in edge_counts.items() if count == 1]


def _extract_boundary_edges_from_lines(line_cells):
    edges = set()
    for cell in line_cells:
        if len(cell) < 2:
            continue
        for i in range(len(cell) - 1):
            a = int(cell[i])
            b = int(cell[i + 1])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edges.add(key)
    return sorted(edges)


def _geometry2d_to_geojson3d(geom2d, frame):
    def ring_to_xyz(ring_coords):
        ring = [_uv_to_xyz(u, v, frame) for u, v in list(ring_coords)]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    if geom2d.geom_type == "Polygon":
        rings = [ring_to_xyz(geom2d.exterior.coords)]
        for interior in geom2d.interiors:
            rings.append(ring_to_xyz(interior.coords))
        return {"type": "Polygon", "coordinates": rings}

    if geom2d.geom_type == "MultiPolygon":
        polygons = []
        for poly in geom2d.geoms:
            rings = [ring_to_xyz(poly.exterior.coords)]
            for interior in poly.interiors:
                rings.append(ring_to_xyz(interior.coords))
            polygons.append(rings)
        return {"type": "MultiPolygon", "coordinates": polygons}

    return None


def _transform_geometry_xy(geometry_obj, transformer):
    gtype = geometry_obj.get("type")
    coords = geometry_obj.get("coordinates")

    if gtype == "Polygon":
        out = []
        for ring in coords:
            t_ring = []
            for x, y, z in ring:
                x_t, y_t = transformer.transform(float(x), float(y))
                t_ring.append([x_t, y_t, float(z)])
            out.append(t_ring)
        return {"type": "Polygon", "coordinates": out}

    if gtype == "MultiPolygon":
        out_polys = []
        for poly in coords:
            out_rings = []
            for ring in poly:
                t_ring = []
                for x, y, z in ring:
                    x_t, y_t = transformer.transform(float(x), float(y))
                    t_ring.append([x_t, y_t, float(z)])
                out_rings.append(t_ring)
            out_polys.append(out_rings)
        return {"type": "MultiPolygon", "coordinates": out_polys}

    return None


def _collect_mesh_edges(tile_mesh):
    faces_cells = _parse_vtk_cells(getattr(tile_mesh, "faces", None))
    line_cells = _parse_vtk_cells(getattr(tile_mesh, "lines", None))

    boundary_edges = []
    if faces_cells:
        boundary_edges.extend(_extract_boundary_edges_from_faces(faces_cells))
    if line_cells:
        boundary_edges.extend(_extract_boundary_edges_from_lines(line_cells))

    # Deduplicate undirected edges
    unique_edges = sorted(set(boundary_edges))
    return unique_edges, faces_cells


def _boundary_geometry_from_mesh(tile_mesh):
    if tile_mesh is None or not hasattr(tile_mesh, "points") or tile_mesh.n_points < 3:
        return None, None

    points = np.asarray(tile_mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return None, None

    edges, _ = _collect_mesh_edges(tile_mesh)
    if not edges:
        return None, None

    point_ids = sorted({idx for edge in edges for idx in edge})
    if len(point_ids) < 3:
        return None, None

    points_subset = points[point_ids, :3]
    frame = _best_fit_plane(points_subset)
    if frame is None:
        return None, None

    uv_all = _to_uv(points[:, :3], frame)
    segments = []
    for a, b in edges:
        pa = (float(uv_all[a, 0]), float(uv_all[a, 1]))
        pb = (float(uv_all[b, 0]), float(uv_all[b, 1]))
        if pa == pb:
            continue
        segments.append(LineString([pa, pb]))

    if not segments:
        return None, None

    polygons = list(polygonize(MultiLineString(segments)))
    if not polygons:
        return None, None

    merged = unary_union(polygons)
    if merged.is_empty:
        return None, None

    if isinstance(merged, GeometryCollection):
        polys = [g for g in merged.geoms if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty]
        if not polys:
            return None, None
        merged = unary_union(polys)
        if merged.is_empty:
            return None, None

    if merged.geom_type not in ("Polygon", "MultiPolygon"):
        return None, None
    if float(merged.area) <= 1e-9:
        return None, None

    return merged, frame


def _triangulation_fallback_geometry(tile_mesh):
    if tile_mesh is None or not hasattr(tile_mesh, "points") or tile_mesh.n_points < 3:
        return None, None
    points = np.asarray(tile_mesh.points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        return None, None

    frame = _best_fit_plane(points[:, :3])
    if frame is None:
        return None, None
    uv_all = _to_uv(points[:, :3], frame)

    edges, faces_cells = _collect_mesh_edges(tile_mesh)
    triangles = []
    for cell in faces_cells:
        if len(cell) < 3:
            continue
        coords_uv = [(float(uv_all[idx, 0]), float(uv_all[idx, 1])) for idx in cell]
        poly = Polygon(coords_uv)
        if poly.is_empty or float(poly.area) <= 1e-9:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or float(poly.area) <= 1e-9:
            continue
        for tri in triangulate(poly):
            if tri.is_empty or float(tri.area) <= 1e-9:
                continue
            overlap = tri.intersection(poly).area
            if overlap <= 1e-9 or overlap / tri.area < 0.99:
                continue
            triangles.append(tri)

    # If face-based triangulation is unavailable (e.g., slice only has polylines),
    # triangulate polygonized boundary lines in projected 2D space.
    if not triangles and edges:
        segments = []
        for a, b in edges:
            pa = (float(uv_all[a, 0]), float(uv_all[a, 1]))
            pb = (float(uv_all[b, 0]), float(uv_all[b, 1]))
            if pa == pb:
                continue
            segments.append(LineString([pa, pb]))
        if segments:
            for poly in polygonize(MultiLineString(segments)):
                if poly.is_empty or float(poly.area) <= 1e-9:
                    continue
                for tri in triangulate(poly):
                    if tri.is_empty or float(tri.area) <= 1e-9:
                        continue
                    overlap = tri.intersection(poly).area
                    if overlap <= 1e-9 or overlap / tri.area < 0.99:
                        continue
                    triangles.append(tri)

    if not triangles:
        return None, None

    merged = unary_union(triangles)
    if merged.is_empty:
        return None, None
    if merged.geom_type not in ("Polygon", "MultiPolygon"):
        polys = []
        if hasattr(merged, "geoms"):
            polys = [g for g in merged.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if not polys:
            return None, None
        merged = MultiPolygon(polys) if len(polys) > 1 else polys[0]
    if float(merged.area) <= 1e-9:
        return None, None
    return merged, frame


def _polygon_ring_area_3d(ring_3d):
    """
    Compute planar polygon area in 3D from a closed ring using Newell's method.
    """
    if ring_3d is None or len(ring_3d) < 4:
        return None

    pts = np.asarray(ring_3d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return None
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    cross_sum = np.zeros(3, dtype=float)
    for i in range(len(pts) - 1):
        cross_sum += np.cross(pts[i], pts[i + 1])

    area = 0.5 * float(np.linalg.norm(cross_sum))
    if not np.isfinite(area):
        return None
    return area


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
    output_path = os.path.abspath(output_path)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    tr = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    features = []
    export_stats = {
        "direct_boundary": 0,
        "triangulation_fallback": 0,
        "skipped": 0,
    }

    for entry in matched_data:
        building_id = _entry_building_id(entry, fallback="unknown")
        intersections = entry.get("face_data", {}).get("intersections", {})
        selected_faces = entry.get("face_data", {}).get("selected_faces", {})
        roof_counter = 1

        for local_face, tile_obj in intersections.items():
            tile_mesh = tile_obj
            if isinstance(tile_obj, (tuple, list)) and len(tile_obj) > 0:
                tile_mesh = tile_obj[0]

            if tile_mesh is None or not hasattr(tile_mesh, "n_points") or tile_mesh.n_points < 3:
                continue

            geom2d, frame = _boundary_geometry_from_mesh(tile_mesh)
            method = "direct_boundary"
            if geom2d is None or frame is None:
                geom2d, frame = _triangulation_fallback_geometry(tile_mesh)
                method = "triangulation_fallback"
                if geom2d is None or frame is None:
                    export_stats["skipped"] += 1
                    print(
                        f"[warn] Skipping roof export for building {building_id} face {local_face}: "
                        "failed boundary reconstruction and fallback triangulation."
                    )
                    continue
                print(
                    f"[warn] Using triangulation fallback for building {building_id} face {local_face}."
                )

            roof_area_m2 = float(geom2d.area)
            geometry_src = _geometry2d_to_geojson3d(geom2d, frame)
            if geometry_src is None:
                export_stats["skipped"] += 1
                print(
                    f"[warn] Skipping roof export for building {building_id} face {local_face}: "
                    "unable to convert reconstructed geometry to GeoJSON."
                )
                continue
            geometry_dst = _transform_geometry_xy(geometry_src, tr)
            if geometry_dst is None:
                export_stats["skipped"] += 1
                print(
                    f"[warn] Skipping roof export for building {building_id} face {local_face}: "
                    "unable to transform geometry coordinates."
                )
                continue
            export_stats[method] += 1

            selected_face = selected_faces.get(str(local_face), {}) if isinstance(selected_faces, dict) else {}
            roof_confidence = selected_face.get("probability") if isinstance(selected_face, dict) else None
            try:
                roof_confidence = None if roof_confidence is None else float(roof_confidence)
            except (TypeError, ValueError):
                roof_confidence = None

            roof_face_global = selected_face.get("global_face") if isinstance(selected_face, dict) else None
            roof_face_global = str(roof_face_global).strip() if roof_face_global is not None else None
            if roof_face_global == "":
                roof_face_global = None

            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "building": building_id,
                        "roof_id": str(roof_counter),
                        "roof_face_local": str(local_face),
                        "roof_face_global": roof_face_global,
                        "roof_confidence": roof_confidence,
                        "roof_area_m2": roof_area_m2,
                    },
                    "geometry": {
                        "type": geometry_dst["type"],
                        "coordinates": geometry_dst["coordinates"],
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

    print(
        f"[ok] Roof surfaces exported to: {output_path} ({len(features)} feature(s)) | "
        f"direct-boundary={export_stats['direct_boundary']} "
        f"triangulation-fallback={export_stats['triangulation_fallback']} "
        f"skipped={export_stats['skipped']}"
    )
    return geojson

_model_cache = None


def get_cached_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = CachedModel()
    return _model_cache

def use_this_function(
    jsonbox,
    building_threshold,
    overlap_threshold,
    output_path="roof_surfaces.geojson",
    zone_shp_path=None,
    polygon_ring_lon_lat=None,
):

    bbox  = json.loads(jsonbox)
    north = bbox["north"]; south = bbox["south"]
    east  = bbox["east"];  west  = bbox["west"]

    log_lines = []
    def log(msg):
        log_lines.append(msg)
        return "\n".join(log_lines)

    # Stage 1 â€” CRS
    print("Converting coordinatesâ€¦")
    t = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True)
    tl_x, tl_y = t.transform(west, north)
    br_x, br_y = t.transform(east, south)
    area_m2 = abs(br_x - tl_x) * abs(tl_y - br_y)
    print(f"ðŸ“  Area: {area_m2/1e6:.4f} kmÂ²")
    print(f"ðŸ”  EPSG:4326 â†’ EPSG:3763 done")


    # Stage 2 â€” OSM
    print("Fetching OSM buildingsâ€¦")
    print("ðŸ—º   Querying Overpass APIâ€¦")
    geojson = get_osm_buildings((tl_x, tl_y), (br_x, br_y))
    print(f"âœ…  OSM done â€” {len(geojson['features'])} footprint(s)")

    # Stage 3 â€” WMTS connect
    print("Connecting to WMTSâ€¦")
    print("ðŸ“¡  Connecting to DGT WMTS satellite serviceâ€¦")
    wmts_url = ("https://cartografia.dgterritorio.gov.pt/ortos2018/service"
                "?service=WMTS&request=GetCapabilities")
    wmts   = WebMapTileService(wmts_url)
    matrix = wmts.tilematrixsets["PTTM_06"].tilematrix["14"]
    col_min, col_max, row_min, row_max = get_tile_indices(tl_x, br_y, br_x, tl_y, matrix)
    total_tiles = (col_max+1-col_min) * (row_max+1-row_min)
    print(f"ðŸ›°   Service ready â€” {total_tiles} tile(s) to download")

    # Stage 4 â€” Tiles
    def tile_cb(done, total):
        print(f"Downloading tilesâ€¦ {done}/{total}")

    satellite_image, res, _ = retrieve_satelite_image(
        (tl_x, tl_y), (br_x, br_y), progress_cb=tile_cb)
    h, w = satellite_image.shape[:2]

    # Stage 5 â€” YOLO
    print("Running YOLO inferenceâ€¦")
    print(f"ðŸ¤–  Running detector  (confâ‰¥{building_threshold:.2f}, iouâ‰¤{overlap_threshold:.2f})â€¦")
    model = get_cached_model()
    img_bgr   = cv2.cvtColor(satellite_image, cv2.COLOR_RGB2BGR)
    buildings = retrieve_prediction_list(img_bgr, (tl_x, tl_y), res,
                                            building_threshold, overlap_threshold, model)
    print(f"âœ…  {len(buildings)} building(s)")

    from PIL import Image
    annotated_image = draw_azimuths_on_satellite(satellite_image, buildings, (tl_x, tl_y), res)
    Image.fromarray(satellite_image).save("satellite_image.png")
    output_path_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        scenario_satellite_raw = os.path.join(output_dir, "satellite_image_raw.png")
        Image.fromarray(satellite_image).save(scenario_satellite_raw)
        print(f"[ok] Saved scenario satellite image: {scenario_satellite_raw}")
    print(f"âœ…  Satellite image ready  ({w}Ã—{h} px, {res:.3f} m/px)")
    # Matching source for IDs: authoritative scenario zone.shp
    if polygon_ring_lon_lat is not None:
        selection_ring = _normalise_polygon_ring(polygon_ring_lon_lat)
    else:
        selection_ring = [
            [west, north],
            [east, north],
            [east, south],
            [west, south],
            [west, north],
        ]
    print("Fetching scenario zone.shp buildings for matching...")
    zone_geojson = build_cea_ordered_buildings_geojson(selection_ring, zone_shp_path=zone_shp_path)
    print(f"Zone matching source loaded: {len(zone_geojson['features'])} footprint(s)")

    print(f"Computing IOU matrix of size [{len(zone_geojson['features'])},{len(buildings)}]")
    iou_mat = compute_iou_matrix(zone_geojson['features'], buildings)
    print("Done computing IOU matrix!")
    matched_data = []
    n_features = len(zone_geojson['features'])
    for i, zone_feat in enumerate(zone_geojson['features']):
        building_id = str(
            zone_feat.get("properties", {}).get("cea_name")
            or zone_feat.get("properties", {}).get("name")
            or zone_feat.get("properties", {}).get("osm_id")
            or ""
        ).strip()
        if not building_id:
            building_id = f"building_{i}"

        best_match_idx = int(np.argmax(iou_mat[i, :]))
        if iou_mat[i, best_match_idx] > 0.3:
            pred = buildings[best_match_idx]
            outline, orientedbox, lines_world, code, face_data = topology_converter_mine(
                pred.raw_roof_data,
                zone_feat,
            )

            matched_data.append(
                {
                    "building_id": building_id,
                    "cea_name": building_id,
                    "osm_id": building_id,
                    "footprint": outline,
                    "roof_planes": orientedbox,
                    "lines_world": lines_world,
                    "code": code,
                    "face_data": face_data,
                }
            )
        if i % 10 == 0 or i == n_features - 1:
            print(f"Matching {i + 1}/{n_features} - {len(matched_data)} matched so far")

    export_roof_surfaces_geojson(matched_data, output_path=output_path)

    maps_output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(maps_output_dir, exist_ok=True)

    print("Rendering map...")
    building_map_html = build_map_html(
        geojson,
        buildings,
        satellite_image,
        (tl_x, tl_y),
        (br_x, br_y),
        matched_data=matched_data,
    )
    building_map_path = os.path.join(maps_output_dir, "building_map.html")
    with open(building_map_path, "w", encoding="utf-8") as f:
        f.write(building_map_html)
    print(f"[ok] Building map written: {building_map_path}")

    roofs_3d_path = os.path.join(maps_output_dir, "3d_roofs.html")
    render_pydeck(geojson, matched_data, out_path=roofs_3d_path)
    print(f"[ok] 3D roofs map written: {roofs_3d_path}")

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

def run_from_polygon_ring(
    polygon_ring_lon_lat,
    building_threshold=0.15,
    overlap_threshold=0.25,
    output_path="roof_surfaces.geojson",
    zone_shp_path=None,
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
        output_path=output_path,
        zone_shp_path=zone_shp_path,
        polygon_ring_lon_lat=ring,
    )

def _build_cli_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run fixedboxtrick from CLI polygon input. "
            "If no source args are given, the script prompts for GeoJSON text."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        "--polygon-ring",
        help=(
            "Polygon coordinates as JSON ring [[lon,lat],...] or GeoJSON text "
            "(FeatureCollection, Feature, Polygon, MultiPolygon, or LineString)."
        ),
    )
    source_group.add_argument(
        "--polygon-geojson",
        help="GeoJSON string describing the polygon area.",
    )
    source_group.add_argument(
        "--polygon-geojson-file",
        help="Path to a .geojson/.json file containing the polygon area.",
    )
    source_group.add_argument(
        "--bbox-json",
        help=(
            "Direct bbox JSON with north/south/east/west "
            "(alternative to polygon/GeoJSON inputs)."
        ),
    )
    parser.add_argument(
        "--zone-shp-path",
        default=None,
        help="Path to scenario zone.shp used for authoritative CEA building ids.",
    )
    parser.add_argument(
        "--output-path",
        default="roof_surfaces.geojson",
        help="Path to write roof surfaces GeoJSON output.",
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
    return parser

def _prompt_polygon_text_from_stdin():
    print("No input arguments provided.")
    print("Paste GeoJSON (or polygon ring JSON).")
    print("Multi-line input is supported; run starts automatically once valid JSON is complete.")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break

        lines.append(line)
        candidate = "\n".join(lines).strip()
        if not candidate:
            continue
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    candidate = "\n".join(lines).strip()
    if not candidate:
        raise ValueError("GeoJSON input is required when running without arguments.")
    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON input. Paste a complete GeoJSON object (or polygon ring JSON)."
        ) from exc
    return candidate

def main():
    parser = _build_cli_parser()
    args = parser.parse_args()

    if not args.zone_shp_path:
        raise ValueError(
            "--zone-shp-path is required. "
            "Matching now uses scenario zone.shp with CEA building names."
        )

    polygon_text = None
    if args.polygon_geojson:
        polygon_text = args.polygon_geojson
    elif args.polygon_geojson_file:
        polygon_text = _read_text_file(args.polygon_geojson_file)
    elif args.polygon_ring:
        polygon_text = args.polygon_ring

    if polygon_text is not None:
        polygon_ring_lon_lat = parse_polygon_text_to_ring(polygon_text)
        run_from_polygon_ring(
            polygon_ring_lon_lat=polygon_ring_lon_lat,
            building_threshold=args.building_threshold,
            overlap_threshold=args.overlap_threshold,
            output_path=args.output_path,
            zone_shp_path=args.zone_shp_path,
        )
        return

    if args.bbox_json:
        use_this_function(
            jsonbox=args.bbox_json,
            building_threshold=args.building_threshold,
            overlap_threshold=args.overlap_threshold,
            output_path=args.output_path,
            zone_shp_path=args.zone_shp_path,
        )
        return

    polygon_text = _prompt_polygon_text_from_stdin()
    polygon_ring_lon_lat = parse_polygon_text_to_ring(polygon_text)
    run_from_polygon_ring(
        polygon_ring_lon_lat=polygon_ring_lon_lat,
        building_threshold=args.building_threshold,
        overlap_threshold=args.overlap_threshold,
        output_path=args.output_path,
        zone_shp_path=args.zone_shp_path,
    )


if __name__ == "__main__":
    main()


