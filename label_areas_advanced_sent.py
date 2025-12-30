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
import math
import threading
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SERVICE_ACCOUNT_FILE = os.getenv("GEE_SERVICE_ACCOUNT_KEY_FILE")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

INPUT_GEOJSON = 'input/archive/input_7.geojson'
OUTPUT_GEOJSON = 'output/centerpoints_sentinel_7.geojson'
GEE_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
IMAGE_DIMENSION = 1024 
METERS_PER_PIXEL = 1.0 
ROI_RADIUS = (IMAGE_DIMENSION * METERS_PER_PIXEL) / 2 

# --- Initialization ---
def initialize_services():
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY)

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
    except Exception as e:
        print(f"Failed to initialize Google Earth Engine: {e}")
        exit(1)

    return genai.GenerativeModel('gemini-3-pro-preview')

# --- Helper Functions ---
def preprocess_image(image):
    """
    Draws a simple RED DOT at the center. 
    """
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size
    cx, cy = width // 2, height // 2
    
    r = 6 
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(255, 0, 0, 255), outline="black")
    
    return img_draw

# --- Core Logic ---
def get_centerpoints(client, images, num_time_steps=0, feature_id="unknown"):
    center_x, center_y = 512, 512

    prompt = (
        "Role: Expert Satellite Image Analyst.\n"
        "Task: Identify the irrigation pivot arm in these NDWI images.\n"
        f"The pivot CENTER is marked with a RED DOT at x={center_x}, y={center_y}.\n"
        "\n**INSTRUCTIONS:**\n"
        "1. **Analyze Each Image in Isolation**: Do not compare images. Look at one image at a time.\n"
        "2. **Find the Radial Line**: Look for a straight line extending outward from the RED DOT.\n"
        "3. **Visual Cues**: The arm appears as a linear feature cutting through the crop texture, a .\n"
        "4. **Empty Fields**: If the field is bare (no crop contrast), set `\"visible\": false`.\n"
        "\n**Output Format:**\n"
        "Return a JSON list containing a SINGLE object.\n"
        "```json\n"
        "[\n"
        "  {\n"
        f"    \"center_x\": {center_x}, \"center_y\": {center_y},\n"
        "    \"arms\": [\n"
        "      {\"image_index\": 0, \"tip_x\": 800, \"tip_y\": 200, \"visible\": true, \"confidence\": 0.95},\n"
        "      {\"image_index\": 1, \"tip_x\": 0, \"tip_y\": 0, \"visible\": false, \"confidence\": 0.0}, \n"
        "      ... \n"
        "    ],\n"
        "    \"thinking\": \"Image 0: Line visible extending East. Image 1: Field is empty, no line visible.\"\n"
        "  }\n"
        "]\n"
        "```\n"
    )

    max_retries = 3
    base_delay = 5

    print("\n--- GEMINI PROMPT ---")
    print(prompt[:600] + "...") 
    print("---------------------\n")

    for attempt in range(max_retries):
        try:
            content = [prompt] + images
            response = client.generate_content(content)
            break 
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(base_delay * (2 ** attempt))
            else:
                print(f"Error communicating with Gemini: {e}")
                return [], None
    else:
        return [], None
        
    text = response.text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('['), text.rfind(']')
        if start != -1 and end != -1:
            data = json.loads(text[start:end+1])
        else:
            return [], None

    if isinstance(data, list):
        return [item for item in data if 'center_x' in item], response.usage_metadata
    return [], None

def get_gee_metadata(roi_geom, start_date='2022-01-01', end_date='2025-12-30'):
    try:
        collection = ee.ImageCollection(GEE_COLLECTION)\
            .filterBounds(roi_geom)\
            .filterDate(start_date, end_date)\
            .filter(ee.Filter.calendarRange(5, 9, 'month'))\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))\
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
                }
                if props.get('system:time_start'):
                    dt = datetime.fromtimestamp(props['system:time_start'] / 1000, timezone.utc)
                    meta['date'] = dt.strftime('%Y-%m-%d')
                    meta['dt_object'] = dt
                metadata_list.append(meta)
        
        selected_images = []
        last_dt = None
        for meta in metadata_list:
            current_dt = meta.get('dt_object')
            if not current_dt: continue
            if last_dt is None or abs((last_dt - current_dt).days) >= 30:
                selected_images.append(meta)
                last_dt = current_dt
            if len(selected_images) >= 5: break 
                
        print(f"    Selected {len(selected_images)} images.")
        for img in selected_images: img.pop('dt_object', None)
        return selected_images
    except Exception as e:
        print(f"Error fetching GEE metadata: {e}")
        return []

def download_images_pair(meta, roi_geom):
    """
    Downloads NDWI using Full Dynamic Range (-1 to 1).
    This creates the lowest possible contrast (softest image).
    """
    try:
        img_id = meta['id']
        ee_img = ee.Image(img_id)
        
        common_params = {
            'dimensions': f"{IMAGE_DIMENSION}x{IMAGE_DIMENSION}",
            'region': roi_geom, 
            'crs': 'EPSG:3857', 
            'format': 'png'
        }

        # Calculate NDWI
        index_val = ee_img.normalizedDifference(['B3', 'B8'])
        
        # --- LOW CONTRAST VISUALIZATION ---
        # Using the full possible range (-1 to 1) prevents color clamping.
        # This renders a very soft, natural looking NDWI map.
        vis_params = {
            'min': -1.0,
            'max': 1.0,
            'palette': ['008000', 'FFFFFF', '0000CC'] # Green -> White -> Blue
        }
        
        final_vis = index_val.visualize(**vis_params)
        
        url_ndwi = final_vis.getThumbURL(common_params)
        resp_ndwi = requests.get(url_ndwi, timeout=30)
        img_ndwi = Image.open(BytesIO(resp_ndwi.content)) if resp_ndwi.status_code == 200 else None
        
        return img_ndwi
    except Exception as e:
        print(f"Error downloading {meta.get('id')}: {e}")
        return None

def process_feature(feature, i, client, file_lock):
    center_lon = feature.get('center_lon')
    center_lat = feature.get('center_lat')
    
    center_point = ee.Geometry.Point([center_lon, center_lat])
    square_roi = center_point.buffer(ROI_RADIUS).bounds()
    roi_info = square_roi.getInfo()
    roi_coords = roi_info['coordinates'][0] 
    lons = [c[0] for c in roi_coords]
    lats = [c[1] for c in roi_coords]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    metadata_list = get_gee_metadata(square_roi)
    if not metadata_list: return

    images_dir = "output/images"
    os.makedirs(images_dir, exist_ok=True)
    
    images = []
    valid_metas = []
    for idx, meta in enumerate(metadata_list):
        img_ndwi = download_images_pair(meta, square_roi)
        
        if img_ndwi:
            img_processed = preprocess_image(img_ndwi)
            images.append(img_processed)
            valid_metas.append(meta)
            
            try:
                img_path = os.path.join(images_dir, f"{feature['id']}_{meta['date']}_{idx}_NDWI.png")
                img_processed.save(img_path)
            except: pass
            
    if not images: return

    detections, usage = get_centerpoints(client, images, len(valid_metas), feature['id'])
    
    if not detections: return

    d = detections[0]
    
    output_features = []
    
    output_features.append({
        "type": "Feature",
        "geometry": { "type": "Point", "coordinates": [center_lon, center_lat] },
        "properties": {
            "type": "Pivot Center",
            "confidence": d.get('confidence', 0.0),
            "thinking": d.get('thinking', ''),
            "processing_date": datetime.now(timezone.utc).isoformat()
        }
    })

    for arm in d.get('arms', []):
        if arm.get('visible') is True:
            idx = arm.get('image_index')
            if idx >= len(valid_metas): continue

            tip_lon = min_lon + (arm['tip_x'] / IMAGE_DIMENSION) * (max_lon - min_lon)
            tip_lat = max_lat - (arm['tip_y'] / IMAGE_DIMENSION) * (max_lat - min_lat)
            
            dx = d['center_x'] - arm['tip_x']
            dy = d['center_y'] - arm['tip_y']
            length = math.sqrt(dx*dx + dy*dy) * METERS_PER_PIXEL
            
            output_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[center_lon, center_lat], [tip_lon, tip_lat]]
                },
                "properties": {
                    "type": "Pivot Arm",
                    "image_date": valid_metas[idx]['date'],
                    "length_meters": length,
                    "confidence": arm.get('confidence', 0.0)
                }
            })
    
    print(f"  Result: Found {len(output_features)-1} arm positions.")
    
    with file_lock:
        try:
            with open(OUTPUT_GEOJSON, 'r+') as f:
                data = json.load(f)
                data['features'].extend(output_features)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"Error updating output file: {e}")

def main():
    client = initialize_services()
    file_lock = threading.Lock()
    
    try:
        with open(INPUT_GEOJSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_GEOJSON} not found.")
        return
    
    center_feature = None
    for f in input_data['features']:
        if f['geometry']['type'] == 'Point':
            center_feature = f
            break
            
    if not center_feature:
        print("CRITICAL: Input GeoJSON must contain a Point feature.")
        return

    combined = {
        "id": "site_0",
        "center_lon": center_feature['geometry']['coordinates'][0],
        "center_lat": center_feature['geometry']['coordinates'][1]
    }

    print(f"Processing Center: {combined['center_lon']}, {combined['center_lat']}")

    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)

    process_feature(combined, 0, client, file_lock)
    print(f"Done. Saved to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()