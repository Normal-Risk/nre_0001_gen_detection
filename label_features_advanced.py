import os
import json
import time
import math
import requests
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from google import genai
from google.oauth2 import service_account
from dotenv import load_dotenv
import ee
import concurrent.futures

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEE_SERVICE_ACCOUNT_KEY_FILE = os.getenv("GEE_SERVICE_ACCOUNT_KEY_FILE")
INPUT_GEOJSON = "input/test_input.geojson"
OUTPUT_GEOJSON = "output/output_original_test.geojson"
FEATURE_PROMPT = "irrigation pivot"
# GEE Parameters
GEE_COLLECTION = "USDA/NAIP/DOQQ"
SCALE = 1.0 # Meters per pixel (NAIP is typically 0.6m or 1m)
REGION_BUFFER = 100 # Meters to buffer around the polygon for context

# ---------------------

def setup_gemini():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return None
    return genai.Client(api_key=GOOGLE_API_KEY)

def setup_gee():
    """Initializes Google Earth Engine."""
    try:
        if GEE_SERVICE_ACCOUNT_KEY_FILE and os.path.exists(GEE_SERVICE_ACCOUNT_KEY_FILE):
            print(f"Authenticating with Service Account: {GEE_SERVICE_ACCOUNT_KEY_FILE}")
            credentials = service_account.Credentials.from_service_account_file(
                GEE_SERVICE_ACCOUNT_KEY_FILE,
                scopes=['https://www.googleapis.com/auth/earthengine']
            )
            ee.Initialize(credentials=credentials)
        else:
            print("Authenticating with default credentials (earthengine authenticate)...")
            ee.Initialize()
            
        print("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        if GEE_SERVICE_ACCOUNT_KEY_FILE:
             print(f"Check if {GEE_SERVICE_ACCOUNT_KEY_FILE} is valid and has correct permissions.")
        else:
            print("Please run 'earthengine authenticate' in your terminal if you haven't already.")
        return False

def get_gee_metadata(geometry_coords, feature_id):
    """
    Fetches metadata for all available images for a given geometry from the NAIP collection.
    Returns a list of metadata dictionaries.
    """
    try:
        roi = ee.Geometry.Polygon(geometry_coords)
        
        collection = ee.ImageCollection(GEE_COLLECTION)\
            .filterBounds(roi)\
            .sort('system:time_start', False)

        # Fetch metadata for up to 50 images in one call
        images_info = collection.limit(50).getInfo()
        
        if 'features' not in images_info:
            print(f"No images found for feature {feature_id}")
            return []
            
        count = len(images_info['features'])
        print(f"Found {count} images for feature {feature_id}")
        
        results = []
        for img_info in images_info['features']:
            img_id = img_info['id']
            props = img_info.get('properties', {})
            timestamp = props.get('system:time_start')
            
            date_str = "Unknown"
            if timestamp:
                dt = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                date_str = dt.isoformat()
                
            results.append({
                "id": img_id,
                "date": date_str,
                "timestamp": timestamp
            })
            
        return results

    except Exception as e:
        print(f"Error fetching GEE metadata: {e}")
        return []

def get_feature_coordinates(client, img, feature_name):
    """
    Sends image to Gemini and asks for line coordinates.
    Returns a list of objects with normalized coordinates and confidence.
    """
    try:
        # Ensure image is in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Enhance contrast and brightness
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.5)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    prompt = (
        f"Analyze this satellite image and exhaustively identify ALL linear structures that could be the {feature_name}. "
        "The image might be low contrast or dark, so look very carefully for faint lines. "
        "Do not miss any potential candidates, even if they are faint or partially obscured. "
        "Include faint linear markings, ground tracks, wheel tracks, crop rows, diagonal streaks, or anything that looks like a line. "
        "Even if you think a line is just a ground texture or artifact, LABEL IT ANYWAY. "
        "Do not filter out any linear features based on confidence. If it looks like a line, include it. "
        "For each potential structure, draw a line segment along its length. "
        "Return a list of these line segments. "
        "Output the result as a JSON list of objects, where each object has keys: "
        "'y1', 'x1', 'y2', 'x2' (integers normalized to [0, 1000]) and "
        "'confidence' (float between 0.0 and 1.0 indicating certainty)."
    )

    try:
        response = client.models.generate_content(
            model='gemini-3-pro-preview',
            contents=[prompt, img]
        )
        
        usage = response.usage_metadata
        print(f"    Token stats: Input={usage.prompt_token_count}, Output={usage.candidates_token_count}, Total={usage.total_token_count}")
        
        text = response.text
        
        # Extract JSON from code block
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
        text = text.replace("'", '"')
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                text = text[start:end+1]
                data = json.loads(text)
            else:
                # raise # Don't raise, just return empty if parsing fails
                print("Failed to parse JSON from Gemini response.")
                return []

        if isinstance(data, dict):
            data = [data]
            
        valid_data = []
        for item in data:
            if all(k in item for k in ['y1', 'x1', 'y2', 'x2', 'confidence']):
                valid_data.append(item)
            else:
                pass # print(f"Skipping invalid item: {item}")
        
        return valid_data

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def initialize_output_file():
    """Creates or overwrites the output file with an empty FeatureCollection."""
    empty_collection = {
        "type": "FeatureCollection",
        "features": []
    }
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump(empty_collection, f, indent=2)

def append_features_to_output(new_features):
    """Appends a list of features to the existing GeoJSON file."""
    if not new_features:
        return

    try:
        with open(OUTPUT_GEOJSON, 'r+') as f:
            data = json.load(f)
            data['features'].extend(new_features)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
    except FileNotFoundError:
        # Should not happen if initialized, but handle gracefully
        with open(OUTPUT_GEOJSON, 'w') as f:
            collection = {
                "type": "FeatureCollection",
                "features": new_features
            }
            json.dump(collection, f, indent=2)

def process_single_image(client, meta, region_coords, bounds_info):
    """
    Downloads and processes a single image.
    """
    try:
        img_id = meta['id']
        # print(f"  Processing image: {img_id} ({meta['date']})") 
        
        ee_img = ee.Image(img_id)
        
        vis_params = {
            'min': 0,
            'max': 255,
            'bands': ['R', 'G', 'B'],
            'dimensions': 1024,
            'region': region_coords
        }
        
        url = ee_img.getThumbURL(vis_params)
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"    Failed to download thumbnail {img_id}: {response.status_code}")
            return []
            
        img_data = Image.open(BytesIO(response.content))
        
        detections = get_feature_coordinates(client, img_data, FEATURE_PROMPT)
        if not detections:
            return []
            
        min_lon, max_lon, min_lat, max_lat = bounds_info
        
        current_image_features = []
        for d in detections:
            norm_x1 = d['x1'] / 1000
            norm_y1 = d['y1'] / 1000
            norm_x2 = d['x2'] / 1000
            norm_y2 = d['y2'] / 1000
            
            lon1 = min_lon + norm_x1 * (max_lon - min_lon)
            lat1 = max_lat - norm_y1 * (max_lat - min_lat)
            lon2 = min_lon + norm_x2 * (max_lon - min_lon)
            lat2 = max_lat - norm_y2 * (max_lat - min_lat)
            
            out_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]]
                },
                "properties": {
                    "input_feature_id": meta.get('feature_id', 'unknown'),
                    "confidence": d['confidence'],
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "imagery_source": "Google Earth Engine (USDA/NAIP/DOQQ)",
                    "source_metadata": {
                        "gee_id": img_id,
                        "date": meta['date'],
                        "timestamp": meta['timestamp'],
                        "collection": GEE_COLLECTION,
                        "scale": SCALE
                    }
                }
            }
            current_image_features.append(out_feature)
            
        return current_image_features
        
    except Exception as e:
        print(f"    Error processing image {meta.get('id')}: {e}")
        return []

def process_feature(client, feature):
    """
    Processes a single GeoJSON feature against all available GEE images in parallel.
    Yields a list of detected features for each processed image.
    """
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    feature_id = props.get("id", "unknown")
    
    if geom.get("type") != "Polygon":
        print(f"Skipping non-polygon feature: {feature_id}")
        return

    # Get coordinates for GEE
    coords = geom["coordinates"] 

    # Pre-calculate bounds and region to avoid repeated GEE calls
    try:
        roi = ee.Geometry.Polygon(coords)
        buffered_roi_info = roi.buffer(REGION_BUFFER).bounds().getInfo()
        region_coords = buffered_roi_info['coordinates']
        
        roi_coords_list = region_coords[0]
        lons = [c[0] for c in roi_coords_list]
        lats = [c[1] for c in roi_coords_list]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        bounds_info = (min_lon, max_lon, min_lat, max_lat)
        
    except Exception as e:
        print(f"Error calculating geometry bounds for {feature_id}: {e}")
        return

    # Fetch metadata
    images_metadata = get_gee_metadata(coords, feature_id)
    
    if not images_metadata:
        return
        
    # Add feature_id to metadata
    for meta in images_metadata:
        meta['feature_id'] = feature_id

    # Process in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_meta = {
            executor.submit(process_single_image, client, meta, region_coords, bounds_info): meta 
            for meta in images_metadata
        }
        
        for future in concurrent.futures.as_completed(future_to_meta):
            try:
                result = future.result()
                if result:
                    yield result
            except Exception as e:
                print(f"Error in worker thread: {e}")

def main():
    if not setup_gee():
        return

    client = setup_gemini()
    if not client:
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)

    try:
        with open(INPUT_GEOJSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_GEOJSON} not found.")
        return

    # Initialize output file
    initialize_output_file()
    
    features = input_data.get("features", [])
    print(f"Processing {len(features)} input features...")
    
    total_detected = 0
    
    for feature in features:
        # Iterate through the generator which yields results per image
        for batch_of_features in process_feature(client, feature):
            append_features_to_output(batch_of_features)
            count = len(batch_of_features)
            total_detected += count
            print(f"  Saved {count} detected features to {OUTPUT_GEOJSON}")
    
    print(f"Processing complete. Total detected features: {total_detected}")

if __name__ == "__main__":
    main()
