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

def get_centerpoints(client, img, feature_id="unknown"):
    """
    Sends the image to Gemini to detect pivot centerpoints, arms, and angles.
    """
    prompt = (
        "You are an expert Remote Sensing Analyst specializing in Precision Agriculture.\n"
        "Your task is to identify **ALL** Center Pivot Irrigation systems visible in this image.\n\n"
        "**Analysis Goals:**\n"
        "For EACH pivot system found, determine:\n"
        "1.  **Centerpoint**: The precise pixel coordinates (x, y) of the central pivot structure.\n"
        "2.  **Pivot Arm**: If a pivot arm is visible, identify its tip coordinates (x, y). If not visible, set to null.\n"
        "3.  **Field Coverage**: Estimate the start and end angles of the irrigated sector (0-360 degrees, 0=North, 90=East). "
        "    *   Full circle = [0, 360].\n"
        "    *   Partial circle (e.g., 'Pac-Man' shape) = [Start, End].\n"
        "4.  **Confidence**: A score (0.0 - 1.0) reflecting your certainty.\n\n"
        "**Output Format:**\n"
        "Return a JSON list of objects. Each object represents ONE pivot:\n"
        "```json\n"
        "[\n"
        "  {\n"
        "    \"center_x\": 512, \"center_y\": 512, // Pixel coordinates [0-1024]\n"
        "    \"arm_tip_x\": 800, \"arm_tip_y\": 200, // Null if not visible\n"
        "    \"coverage_angles\": [0, 270], // [Start, End] in degrees\n"
        "    \"confidence\": 0.95,\n"
        "    \"thinking\": \"Step-by-step thought process...\",\n"
        "    \"reasoning\": \"Found clear pivot center. Arm extends NE. Field covers 3/4 circle.\"\n"
        "  }\n"
        "]\n"
        "```\n"
        "If NO pivots are found, return an empty list `[]`."
    )

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.generate_content([prompt, img])
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

def get_gee_metadata(geometry_coords, feature_id, start_date='2020-01-01', end_date='2024-01-01'):
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

def process_single_image(client, meta, region_coords, bounds_info):
    """Downloads image and runs detection."""
    try:
        img_id = meta['id']
        ee_img = ee.Image(img_id)
        vis_params = {
            'min': 0, 'max': 255, 'bands': ['R', 'G', 'B'],
            'dimensions': IMAGE_DIMENSION, 'region': region_coords
        }
        
        url = ee_img.getThumbURL(vis_params)
        response = requests.get(url)
        if response.status_code != 200: return [], None
            
        img_data = Image.open(BytesIO(response.content))
        
        detections, usage = get_centerpoints(client, img_data, meta.get('id', 'unknown'))
        if not detections: return [], usage

        if usage:
            print(f"    Token Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count}")

        min_lon, max_lon, min_lat, max_lat = bounds_info
        
        features = []
        for d in detections:
            # Convert Center Pixel to Lat/Lon
            norm_cx = d['center_x'] / IMAGE_DIMENSION
            norm_cy = d['center_y'] / IMAGE_DIMENSION
            
            center_lon = min_lon + norm_cx * (max_lon - min_lon)
            center_lat = max_lat - norm_cy * (max_lat - min_lat) # Y is inverted

            # Convert Arm Tip Pixel to Lat/Lon (if exists)
            arm_geometry = None
            if d.get('arm_tip_x') is not None and d.get('arm_tip_y') is not None:
                norm_tx = d['arm_tip_x'] / IMAGE_DIMENSION
                norm_ty = d['arm_tip_y'] / IMAGE_DIMENSION
                tip_lon = min_lon + norm_tx * (max_lon - min_lon)
                tip_lat = max_lat - norm_ty * (max_lat - min_lat)
                
                arm_geometry = {
                    "type": "LineString",
                    "coordinates": [[center_lon, center_lat], [tip_lon, tip_lat]]
                }

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
                    "pivot_arm": arm_geometry,
                    "length_meters": None,
                    "processing_date": datetime.now(timezone.utc).isoformat(),
                    "imagery_source": "Google Earth Engine (USDA/NAIP/DOQQ)",
                    "source_metadata": meta
                }
            }

            # Calculate Length in Meters
            if d.get('center_x') is not None and d.get('center_y') is not None and \
               d.get('arm_tip_x') is not None and d.get('arm_tip_y') is not None:
                dx = d['center_x'] - d['arm_tip_x']
                dy = d['center_y'] - d['arm_tip_y']
                pixel_dist = math.sqrt(dx*dx + dy*dy)
                feature['properties']['length_meters'] = pixel_dist * SCALE
            features.append(feature)
            
        return features, usage
    except Exception as e:
        print(f"Error processing image {meta.get('id')}: {e}")
        return [], None

def main():
    client = initialize_services()
    
    # Load Input
    try:
        with open(INPUT_GEOJSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_GEOJSON} not found.")
        return
    
    all_features = []
    
    print(f"Processing {len(input_data['features'])} features from {INPUT_GEOJSON}...")
    
    # Initialize Output File
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": []}, f, indent=2)

    for i, feature in enumerate(input_data['features']):
        feature_id = feature.get('id', f'feature_{i}')
        geom = feature['geometry']
        if geom['type'] != 'Polygon': continue
        
        coords = geom['coordinates'] # GEE expects list of rings
        
        # Calculate bounds and buffered region
        try:
            roi = ee.Geometry.Polygon(coords)
            buffered_roi_info = roi.buffer(REGION_BUFFER).bounds().getInfo()
            region_coords = buffered_roi_info['coordinates']
            
            # Calculate bounds for pixel conversion (using buffered region)
            roi_coords_list = region_coords[0]
            lons = [c[0] for c in roi_coords_list]
            lats = [c[1] for c in roi_coords_list]
            bounds_info = (min(lons), max(lons), min(lats), max(lats))
            
        except Exception as e:
            print(f"Error calculating geometry for {feature_id}: {e}")
            continue
        
        # Get Metadata (pass original coords for filtering, or buffered? usually original)
        # label_features_optimized uses original coords for get_gee_metadata
        metadata_list = get_gee_metadata(coords[0], feature_id) 
        
        # Only process the most recent image
        if metadata_list:
            metadata_list = metadata_list[:1] 
        
        if not metadata_list:
            print(f"  No images found for {feature_id}")
            continue

        # Process Images
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_single_image, client, meta, region_coords, bounds_info)
                for meta in metadata_list
            ]
            
            for future in concurrent.futures.as_completed(futures):
                result, usage = future.result()
                if result:
                    all_features.extend(result)
                    print(f"  Found {len(result)} centerpoints for {feature_id}")
                    
                    # Real-time Output Update
                    try:
                        with open(OUTPUT_GEOJSON, 'r+') as f:
                            data = json.load(f)
                            data['features'].extend(result)
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                    except Exception as e:
                        print(f"Error updating output file: {e}")

    print(f"Saved {len(all_features)} centerpoints to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
