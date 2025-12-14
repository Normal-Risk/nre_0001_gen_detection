import os
import json
import time
import requests
import google.generativeai as genai
import ee
from google.oauth2.service_account import Credentials
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime, timezone
import concurrent.futures
import math
import threading
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# Service Account Key File
SERVICE_ACCOUNT_FILE = os.getenv("GEE_SERVICE_ACCOUNT_KEY_FILE")

# Gemini API Key
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Input/Output Files
INPUT_GEOJSON = 'input/archive/input.geojson'
OUTPUT_GEOJSON = 'output/centerpoints_sentinel.geojson'

# GEE Collection
GEE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
SCALE = 1.0 # Meters per pixel (Approximate for the downloaded crop, note Sentinel native is 10m)
REGION_BUFFER = 200 # Meters to buffer around the polygon for context

# Image Parameters
IMAGE_DIMENSION = 1024 


# --- Initialization ---

def initialize_services():
    """Initializes GEE and Gemini."""
    # 1. Initialize Gemini
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)

    # 2. Initialize Earth Engine
    try:
        if os.path.exists(SERVICE_ACCOUNT_FILE):
            print(f"Authenticating with Service Account: {SERVICE_ACCOUNT_FILE}")
            credentials = Credentials.from_service_account_file(
                SERVICE_ACCOUNT_FILE,
                scopes=['https://www.googleapis.com/auth/earthengine']
            )
            ee.Initialize(credentials=credentials)
        else:
            print("Service account key not found. Attempting default authentication...")
            ee.Initialize()
        print("Google Earth Engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")
        exit(1)

    return genai.GenerativeModel('gemini-3-pro-preview')

# --- Core Logic ---

def get_centerpoints(client, images, num_time_steps=0, feature_id="unknown"):
    """
    Sends multiple images to Gemini to detect a consensus centerpoint and individual arms.
    """
    prompt = (
        "You are an expert Remote Sensing Analyst specializing in Precision Agriculture.\n"
        "Your task is to analyze these satellite images of the SAME field taken on different dates.\n"
        "Identify the **Center Pivot Irrigation system** present in the field.\n\n"
        "**Analysis Steps for EACH Time Step:**\n"
        "You are provided with a sequence of image PAIRS. For each date $t$, you get [Image $t$_RGB, Image $t$_NDWI].\n"
        "1.  **Locate Center**: Identify the static central hub. This should be consistent across all images.\n"
        "2.  **Identify Service Road**: Find the bright white static line connecting the center to the edge.\n"
        "3.  **Identify Irrigation Arm**: For **EVERY** time step (image pair), identify the visible pivot arm using BOTH images:\n"
        "    *   **RGB Image**: Shows true color. Useful for spotting the metal structure/truss of the arm.\n"
        "    *   **NDWI Image**: Highlights water (blue) vs vegetation (green). Useful if the arm is irrigating (wet) or contrasting with crop.\n"
        "    *   **Combine Cues**: Use the pair to confirm the arm's presence. Return ONE result per time step.\n"
        f"4.  **Mandatory Output**: You provided {num_time_steps} time steps. You MUST return exactly {num_time_steps} arm detections. Do not skip ANY dates.\n"
        "    *   **Detect FAINT features**: Look for very subtle, low-contrast lines, 'ghost lines' in RGB.\n"
        "3.  **Field Coverage**: Estimate the irrigated sector angles.\n\n"
        "**Output Format:**\n"
        "Return a JSON list containing a SINGLE object representing the pivot system. The 'arms' list MUST have the same number of entries as the input images.\n"
        "```json\n"
        "[\n"
        "  {\n"
        "    \"center_x\": 512, \"center_y\": 512, // Static center pixel [0-1024]\n"
        "    \"arms\": [\n"
        "      {\"image_index\": 0, \"tip_x\": 800, \"tip_y\": 200, \"visible\": true, \"confidence\": 0.95},\n"
        "      {\"image_index\": 1, \"tip_x\": 400, \"tip_y\": 600, \"visible\": true, \"confidence\": 0.40},\n"
        "      ... // Must include an entry for every image index\n"
        "    ],\n"
        "    \"coverage_angles\": [0, 270],\n"
        "    \"confidence\": 0.95,\n"
        "    \"thinking\": \"Step-by-step thought process...\",\n"
        "    \"reasoning\": \"Located static center. Traced arm in Image 1 extending NE. Image 2 arm extends S.\"\n"
        "  }\n"
        "]\n"
        "```\n"
        "If NO pivot is found, return `[]`."
    )

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            # Pass the prompt and ALL images
            content = [prompt] + images
            response = client.generate_content(content)
            break # Success
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt)
                print(f"    Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Error communicating with Gemini: {e}")
                return [], None
    else:
        print(f"    Failed after {max_retries} retries.")
        return [], None
        
    text = response.text
    # Clean up code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end]
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find list brackets
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end+1]
            data = json.loads(text)
        else:
            return [], None

    if not isinstance(data, list):
        return [], None

    valid_data = []
    for item in data:
        if 'center_x' in item and 'center_y' in item:
            valid_data.append(item)
    
    return valid_data, response.usage_metadata

def get_gee_metadata(geometry_coords, feature_id, start_date='2010-01-01', end_date='2030-01-01'):
    """Fetches metadata for available NAIP images."""
    try:
        roi = ee.Geometry.Polygon(geometry_coords)
        collection = ee.ImageCollection(GEE_COLLECTION)\
            .filterBounds(roi)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .sort('system:time_start', False)
            
        images_info = collection.getInfo()
        
        metadata_list = []
        if 'features' in images_info:
            for img in images_info['features']:
                props = img.get('properties', {})
                meta = {
                    'id': img['id'],
                    'date': props.get('system:index', '')[:8],
                    'timestamp': props.get('system:time_start'),
                    'all_properties': props
                }
                if props.get('system:time_start'):
                    dt = datetime.fromtimestamp(props['system:time_start'] / 1000, timezone.utc)
                    meta['date'] = dt.strftime('%Y-%m-%d')
                    meta['dt_object'] = dt # Store datetime object for easier comparison
                metadata_list.append(meta)
        
        # Filter: Max 10 images, spaced by at least 30 days
        selected_images = []
        last_dt = None
        
        for meta in metadata_list:
            current_dt = meta.get('dt_object')
            if not current_dt: continue
            
            if last_dt is None:
                selected_images.append(meta)
                last_dt = current_dt
            else:
                # Calculate difference in days
                diff = abs((last_dt - current_dt).days)
                if diff >= 30:
                    selected_images.append(meta)
                    last_dt = current_dt
            
            if len(selected_images) >= 10:
                break
                
        print(f"    Selected {len(selected_images)} images after filtering (max 10, >30 days gap).")
        # Cleanup temporary datetime objects to avoid JSON serialization errors
        for img in selected_images:
            img.pop('dt_object', None)
        return selected_images
    except Exception as e:
        print(f"Error fetching GEE metadata: {e}")
        return []

def download_images_pair(meta, region_coords):
    """Downloads BOTH RGB and NDWI images from GEE for a given metadata entry."""
    try:
        img_id = meta['id']
        ee_img = ee.Image(img_id)
        
        # --- 1. RGB Visualization ---
        # Sentinel-2: B4 (Red), B3 (Green), B2 (Blue)
        vis_params_rgb = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 2800, # Slightly brighter to catch faint lines
            'gamma': 1.4,
            'dimensions': IMAGE_DIMENSION, 
            'region': region_coords,
            'crs': 'EPSG:4326',
            'format': 'png'
        }
        url_rgb = ee_img.getThumbURL(vis_params_rgb)
        resp_rgb = requests.get(url_rgb, timeout=30)
        img_rgb = Image.open(BytesIO(resp_rgb.content)) if resp_rgb.status_code == 200 else None

        # --- 2. NDWI Visualization ---
        # Calculate Index (B3-B8)/(B3+B8)
        index_val = ee_img.normalizedDifference(['B3', 'B8'])
        
        # Viz 1: Green for Veg (<0)
        neg_mask = index_val.lt(0)
        neg_input = index_val.multiply(-1).max(0).min(1).mask(neg_mask)
        vis_veg = neg_input.visualize(min=0, max=1, palette=['FFFFFF', '008000'])
        
        # Viz 2: Blue for Water (>=0)
        pos_mask = index_val.gte(0)
        pos_input = index_val.mask(pos_mask).sqrt().sqrt().max(0).min(1)
        vis_water = pos_input.visualize(min=0, max=1, palette=['FFFFFF', '0000CC'])
        
        # Blend
        final_vis_ndwi = ee.ImageCollection([vis_veg, vis_water]).mosaic()
        
        vis_params_ndwi = {
            'dimensions': IMAGE_DIMENSION, 
            'region': region_coords,
            'crs': 'EPSG:4326',
            'format': 'png'
        }
        url_ndwi = final_vis_ndwi.getThumbURL(vis_params_ndwi)
        resp_ndwi = requests.get(url_ndwi, timeout=30)
        img_ndwi = Image.open(BytesIO(resp_ndwi.content)) if resp_ndwi.status_code == 200 else None
        
        return img_rgb, img_ndwi
    except Exception as e:
        print(f"Error downloading images for {meta.get('id')}: {e}")
        return None, None

def process_feature(feature, i, client, file_lock):
    """Processes a single feature using multiple images."""
    feature_id = feature.get('id', f'feature_{i}')
    known_center_lon = feature.get('center_lon')
    known_center_lat = feature.get('center_lat')
    
    # Use the combined geometry (LineString, Polygon, etc) for the ROI
    roi_geom_type = feature['geometry']['type']
    roi_coords = feature['geometry']['coordinates']

    if roi_geom_type == 'LineString':
        roi = ee.Geometry.LineString(roi_coords)
    elif roi_geom_type == 'Polygon':
        roi = ee.Geometry.Polygon(roi_coords)
    elif roi_geom_type == 'Point':
        roi = ee.Geometry.Point(roi_coords)
    else:
        # Fallback or error
        print(f"Unsupported geometry type for ROI: {roi_geom_type}")
        return

    # Calculate bounds and buffered region
    try:
        buffered_roi_info = roi.buffer(REGION_BUFFER).bounds().getInfo()
        region_coords = buffered_roi_info['coordinates']
        
        roi_coords_list = region_coords[0]
        lons = [c[0] for c in roi_coords_list]
        lats = [c[1] for c in roi_coords_list]
        bounds_info = (min(lons), max(lons), min(lats), max(lats))
        min_lon, max_lon, min_lat, max_lat = bounds_info
        
    except Exception as e:
        print(f"Error calculating geometry for {feature_id}: {e}")
        return
    
    # Get Metadata (All available)
    # Use the ROI (which handles both Point and Polygon) to fetch metadata
    # We pass the coordinates of the ROI buffer to get_gee_metadata if needed, 
    # but the original function expects geometry_coords. 
    # Let's adjust get_gee_metadata slightly or pass the region_coords.
    # Actually get_gee_metadata iterates coords[0] assuming polygon.
    # Let's simplify get_gee_metadata usage. It constructs a Polygon from input.
    # We can pass `region_coords`.
    metadata_list = get_gee_metadata(region_coords, feature_id) 
    # if metadata_list:
    #     metadata_list = metadata_list[:15] 
    
    if not metadata_list:
        print(f"  No images found for {feature_id}")
        return

    # Create output directory for images
    images_dir = "output/images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Download Images
    images = []
    valid_metas = []
    for idx, meta in enumerate(metadata_list):
        img_rgb, img_ndwi = download_images_pair(meta, region_coords)
        
        if img_rgb and img_ndwi:
            # Append BOTH to the flat list
            images.append(img_rgb)
            images.append(img_ndwi)
            valid_metas.append(meta)
            
            # Save Images for Debugging
            try:
                img_path_rgb = os.path.join(images_dir, f"{feature_id}_{meta['date']}_{idx}_RGB.png")
                img_rgb.save(img_path_rgb)
                img_path_ndwi = os.path.join(images_dir, f"{feature_id}_{meta['date']}_{idx}_NDWI.png")
                img_ndwi.save(img_path_ndwi)
            except Exception as e:
                print(f"    Failed to save debug images: {e}")
            
    if not images:
        return

    # Run Detection on Sequence
    num_time_steps = len(valid_metas)
    detections, usage = get_centerpoints(client, images, num_time_steps, feature_id)
    
    if usage:
        print(f"    Token Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count}")

    if not detections:
        return

    # Process Consensus Detection
    d = detections[0] # Expecting single object for the pivot
    
    if known_center_lon is not None and known_center_lat is not None:
        # Use known center from input
        center_lon = known_center_lon
        center_lat = known_center_lat
        
        # Calculate pixel coordinates for the known center to help with arm calculations if needed
        # (Though we largely use the center for output)
        norm_cx = (center_lon - min_lon) / (max_lon - min_lon)
        norm_cy = (max_lat - center_lat) / (max_lat - min_lat)
        # We can update d['center_x'] etc if we wanted to normalize the model's output relative to true center
        # but the request is mainly to OUTPUT the center from input.
    else:
        # Convert Center Pixel to Lat/Lon
        norm_cx = d['center_x'] / IMAGE_DIMENSION
        norm_cy = d['center_y'] / IMAGE_DIMENSION
        
        center_lon = min_lon + norm_cx * (max_lon - min_lon)
        center_lat = max_lat - norm_cy * (max_lat - min_lat)

    # Process Arms
    arms_data = []
    lengths = []
    
    for arm in d.get('arms', []):
        idx = arm.get('image_index')
        if idx is not None and 0 <= idx < len(valid_metas) and arm.get('visible'):
            # Convert Tip Pixel
            norm_tx = arm['tip_x'] / IMAGE_DIMENSION
            norm_ty = arm['tip_y'] / IMAGE_DIMENSION
            tip_lon = min_lon + norm_tx * (max_lon - min_lon)
            tip_lat = max_lat - norm_ty * (max_lat - min_lat)
            
            # Calculate Length
            dx = d['center_x'] - arm['tip_x']
            dy = d['center_y'] - arm['tip_y']
            length = math.sqrt(dx*dx + dy*dy) * SCALE
            lengths.append(length)
            
            arms_data.append({
                "image_date": valid_metas[idx]['date'],
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[center_lon, center_lat], [tip_lon, tip_lat]]
                },
                "length_meters": length,
                "confidence": arm.get('confidence', 0.0)
            })

    avg_length = sum(lengths) / len(lengths) if lengths else None

    # Construct Feature
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [center_lon, center_lat]
        },
        "properties": {
            "confidence": d.get('confidence', 0.0),
            "thinking": d.get('thinking', ''),
            "reasoning": d.get('reasoning', ''),
            "field_coverage": d.get('coverage_angles', []),
            "avg_length_meters": avg_length,
            "arms": arms_data,
            "processing_date": datetime.now(timezone.utc).isoformat(),
            "imagery_source": "COPERNICUS/S2_SR_HARMONIZED",
            "source_metadata": valid_metas
        }
    }
    
    print(f"  Found consensus centerpoint for {feature_id} with {len(arms_data)} arms")
    
    # Real-time Output Update (Thread-safe)
    with file_lock:
        try:
            with open(OUTPUT_GEOJSON, 'r+') as f:
                data = json.load(f)
                data['features'].append(feature)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"Error updating output file: {e}")
    return

def main():
    client = initialize_services()
    file_lock = threading.Lock()
    
    # Load Input
    try:
        with open(INPUT_GEOJSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_GEOJSON} not found.")
        return
    
    # Parse Input Data into a Single Site Definition if possible, or process features intelligently
    # The user implied the Input File contains components of ONE site (LineString for bounds, Point for center).
    # We will aggregate feature geometries.
    
    input_features = input_data['features']
    if not input_features:
        print("No features found.")
        return

    combined_feature = {
        "id": "site_0",
        "geometry": None,
        "center_lon": None,
        "center_lat": None
    }
    
    # Look for LineString (for bounds) and Point (for center)
    linestring_geom = None
    point_geom = None
    
    for f in input_features:
        g_type = f['geometry']['type']
        if g_type == 'LineString':
            linestring_geom = f['geometry']
        elif g_type == 'Point':
            point_geom = f['geometry']
        elif g_type == 'Polygon':
            linestring_geom = f['geometry'] # Treat polygon as bounds provider too if line missing

    if point_geom:
        combined_feature["center_lon"] = point_geom['coordinates'][0]
        combined_feature["center_lat"] = point_geom['coordinates'][1]
    
    if linestring_geom:
         combined_feature["geometry"] = linestring_geom
    elif point_geom:
         # Fallback to just point if no line
         combined_feature["geometry"] = point_geom
    else:
        print("No valid geometry found (LineString, Polygon, or Point).")
        return

    print(f"Processing consolidated site: Center=({combined_feature['center_lon']}, {combined_feature['center_lat']}), BoundsSource={combined_feature['geometry']['type']}")

    # Ensure output directory exists and initialize file
    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)

    process_feature(combined_feature, 0, client, file_lock)

    print(f"Processing complete. Results saved to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
