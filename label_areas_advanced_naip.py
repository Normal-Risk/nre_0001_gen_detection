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
INPUT_GEOJSON = 'input/input.geojson'
OUTPUT_GEOJSON = 'output/centerpoints.geojson'

# GEE Collection
GEE_COLLECTION = "USDA/NAIP/DOQQ"
SCALE = 1.0 # Meters per pixel
REGION_BUFFER = 100 # Meters to buffer around the polygon for context

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

def get_centerpoints(client, images, feature_id="unknown"):
    """
    Sends multiple images to Gemini to detect a consensus centerpoint and individual arms.
    """
    prompt = (
        "You are an expert Remote Sensing Analyst specializing in Precision Agriculture.\n"
        "Your task is to analyze these satellite images of the SAME field taken on different dates.\n"
        "Identify the **Center Pivot Irrigation system** present in the field.\n\n"
        "**Analysis Goals:**\n"
        "1.  **Consensus Centerpoint**: Identify the **static** physical center of the pivot (concrete pad/tower) that remains constant across all images. Use the multiple views to triangulate the precise location.\n"
        "2.  **Pivot Arms**: For **EVERY** image provided in the sequence, identify the visible pivot arm and its tip coordinates. Do not skip any images.\n"
        "3.  **Field Coverage**: Estimate the irrigated sector angles.\n\n"
        "**Output Format:**\n"
        "Return a JSON list containing a SINGLE object representing the pivot system:\n"
        "```json\n"
        "[\n"
        "  {\n"
        "    \"center_x\": 512, \"center_y\": 512, // Static center pixel [0-1024]\n"
        "    \"arms\": [\n"
        "      {\"image_index\": 0, \"tip_x\": 800, \"tip_y\": 200, \"visible\": true},\n"
        "      {\"image_index\": 1, \"tip_x\": 400, \"tip_y\": 600, \"visible\": true},\n"
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
                metadata_list.append(meta)
        return metadata_list
    except Exception as e:
        print(f"Error fetching GEE metadata: {e}")
        return []

def download_image(meta, region_coords):
    """Downloads a single image from GEE."""
    try:
        img_id = meta['id']
        ee_img = ee.Image(img_id)
        vis_params = {
            'min': 0, 'max': 255, 'bands': ['R', 'G', 'B'],
            'dimensions': IMAGE_DIMENSION, 'region': region_coords,
            'crs': 'EPSG:4326'
        }
        
        url = ee_img.getThumbURL(vis_params)
        response = requests.get(url)
        if response.status_code != 200: return None
            
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image {meta.get('id')}: {e}")
        return None

def process_feature(feature, i, client, file_lock):
    """Processes a single feature using multiple images."""
    feature_id = feature.get('id', f'feature_{i}')
    geom = feature['geometry']
    if geom['type'] != 'Polygon': return
    
    coords = geom['coordinates']
    
    # Calculate bounds and buffered region
    try:
        roi = ee.Geometry.Polygon(coords)
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
    metadata_list = get_gee_metadata(coords[0], feature_id) 
    # if metadata_list:
    #     metadata_list = metadata_list[:15] 
    
    if not metadata_list:
        print(f"  No images found for {feature_id}")
        return

    # Download Images
    images = []
    valid_metas = []
    for meta in metadata_list:
        img = download_image(meta, region_coords)
        if img:
            images.append(img)
            valid_metas.append(meta)
            
    if not images:
        return

    # Run Detection on Sequence
    detections, usage = get_centerpoints(client, images, feature_id)
    
    if usage:
        print(f"    Token Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count}")

    if not detections:
        return

    # Process Consensus Detection
    d = detections[0] # Expecting single object for the pivot
    
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
                "length_meters": length
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
            "imagery_source": "Google Earth Engine (USDA/NAIP/DOQQ)",
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
    
    print(f"Processing {len(input_data['features'])} features from {INPUT_GEOJSON}...")
    
    # Initialize Output File
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)

    # Parallel Processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_feature, feature, i, client, file_lock)
            for i, feature in enumerate(input_data['features'])
        ]
        
        # Wait for all to complete
        concurrent.futures.wait(futures)

    print(f"Processing complete. Results saved to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
