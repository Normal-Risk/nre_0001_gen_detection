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
INPUT_GEOJSON = "input/input.geojson"
OUTPUT_GEOJSON = "output/output_optimized.geojson"
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


def get_feature_lines(client, img, feature_name):
    """
    Sends image to Gemini and asks for line coordinates with reasoning.
    Returns a list of objects with normalized coordinates, confidence, and reasoning.
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
        f"You are an expert Remote Sensing Analyst specializing in Precision Agriculture. "
        f"Your task is to identify the precise location of the {feature_name} (irrigation pivot arm) in this image.\n\n"
        "**Step-by-Step Analysis (Chain of Thought):**\n"
        "1.  **Context**: Identify the circular field pattern typical of center pivot irrigation.\n"
        "2.  **Center**: A YELLOW DOT has been drawn on the image to mark the EXACT center (centroid). The pivot arm STARTS here.\n"
        "3.  **360° Scan**: Imagine standing at the YELLOW DOT. Look 360 degrees around you.\n"
        "    *   Find the ONE direction where a straight, rigid structure extends from the dot to the outer edge.\n"
        "    *   *Visual Cues*: It looks like a long truss or bridge. It is a straight radius.\n"
        "    *   *Differentiation*: It cuts ACROSS the concentric circular wheel tracks. It is NOT a curved line.\n"
        "4.  **Coordinates**: Determine the end point (outer tip) of this arm. The start point is FIXED at the yellow dot.\n\n"
        "**Confidence Scoring Rubric:**\n"
        "*   **0.95 - 1.0**: Arm is CRYSTAL CLEAR, high contrast, continuous, and unambiguously identified.\n"
        "*   **0.80 - 0.94**: Arm is visible but may be faint, low contrast, or partially obscured by crop canopy.\n"
        "*   **< 0.80**: Ambiguous, uncertain, or could be a wheel track. (These will be filtered out).\n\n"
        "**Output Format:**\n"
        "Return a JSON list containing a single object with the following keys:\n"
        "*   `reasoning`: A concise explanation including the COMPASS DIRECTION (e.g., 'Found arm extending North-East from the yellow dot').\n"
        "*   `y1`, `x1`: Start coordinates (Center/Yellow Dot) [0-1000]\n"
        "*   `y2`, `x2`: End coordinates (Outer Tip) [0-1000]\n"
        "*   `confidence`: Confidence score [0.0-1.0] based on the rubric.\n"
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
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
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                text = text[start:end+1]
                data = json.loads(text)
            else:
                print("Failed to parse JSON from Gemini response.")
                return []

        if isinstance(data, dict):
            data = [data]
            
        valid_data = []
        for item in data:
            if all(k in item for k in ['y1', 'x1', 'y2', 'x2', 'confidence']):
                valid_data.append(item)
            else:
                pass 
        
        return valid_data

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def get_gee_metadata(geometry_coords, feature_id, start_date='2020-01-01', end_date='2024-01-01'):
    """
    Fetches metadata for available NAIP images for a given feature.
    Returns a list of dicts with image ID, date, and ALL properties.
    """
    try:
        roi = ee.Geometry.Polygon(geometry_coords)
        
        collection = ee.ImageCollection(GEE_COLLECTION)\
            .filterBounds(roi)\
            .filterDate(start_date, end_date)\
            .sort('system:time_start', False)
            
        # Limit to MAX_IMAGES_PER_FEATURE to save costs
        images_info = collection.getInfo()
        
        metadata_list = []
        if 'features' in images_info:
            for img in images_info['features']:
                props = img.get('properties', {})
                # Ensure critical fields are present at top level for easy access, but keep full props
                meta = {
                    'id': img['id'],
                    'date': props.get('system:index', '')[:8], # Approximate date from index if needed
                    'timestamp': props.get('system:time_start'),
                    'all_properties': props # Store ALL properties here
                }
                # Try to get a better date string if available
                if props.get('system:time_start'):
                    dt = datetime.fromtimestamp(props['system:time_start'] / 1000, timezone.utc)
                    meta['date'] = dt.strftime('%Y-%m-%d')
                    
                metadata_list.append(meta)
                
        return metadata_list
        
    except Exception as e:
        print(f"Error fetching GEE metadata: {e}")
        return []

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

def process_single_image(client, meta, region_coords, bounds_info, centroid_coords):
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
        
        # Draw visual centerpoint hint
        norm_centroid_x = 500 # Default center if calculation fails
        norm_centroid_y = 500
        
        if centroid_coords:
            min_lon, max_lon, min_lat, max_lat = bounds_info
            c_lon, c_lat = centroid_coords
            
            # Convert lat/lon to pixel coordinates (1024x1024)
            # X: (lon - min_lon) / (max_lon - min_lon) * width
            # Y: (max_lat - lat) / (max_lat - min_lat) * height (Y is inverted)
            
            if (max_lon - min_lon) > 0 and (max_lat - min_lat) > 0:
                px = int((c_lon - min_lon) / (max_lon - min_lon) * 1024)
                py = int((max_lat - c_lat) / (max_lat - min_lat) * 1024)
                
                # Calculate normalized coordinates (0-1000) for force-snapping
                norm_centroid_x = (c_lon - min_lon) / (max_lon - min_lon) * 1000
                norm_centroid_y = (max_lat - c_lat) / (max_lat - min_lat) * 1000
                
                # Draw yellow dot
                draw = ImageDraw.Draw(img_data)
                r = 5 # Radius
                draw.ellipse((px-r, py-r, px+r, py+r), fill='yellow', outline='yellow')
        
        detections = get_feature_lines(client, img_data, FEATURE_PROMPT)
        if not detections:
            return []
            
        min_lon, max_lon, min_lat, max_lat = bounds_info
        
        current_image_features = []
        for d in detections:
            # Filter out low confidence detections
            if d['confidence'] < 0.75:
                # print(f"    Skipping low confidence detection: {d['confidence']}")
                continue

            # Merge specific metadata with full GEE properties
            source_meta = {
                "gee_id": img_id,
                "date": meta['date'],
                "timestamp": meta['timestamp'],
                "collection": GEE_COLLECTION,
                "scale": SCALE
            }
            # Add all other GEE properties
            if 'all_properties' in meta:
                source_meta.update(meta['all_properties'])

            # FORCE SNAP: Overwrite start point with calculated centroid
            d['x1'] = norm_centroid_x
            d['y1'] = norm_centroid_y

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
                    "reasoning": d.get('reasoning', 'No reasoning provided'),
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "imagery_source": "Google Earth Engine (USDA/NAIP/DOQQ)",
                    "source_metadata": source_meta
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
    centroid_coords = None
    try:
        roi = ee.Geometry.Polygon(coords)
        buffered_roi_info = roi.buffer(REGION_BUFFER).bounds().getInfo()
        region_coords = buffered_roi_info['coordinates']
        
        # Calculate centroid for visual hint
        centroid_coords = roi.centroid().coordinates().getInfo()
        
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
            executor.submit(process_single_image, client, meta, region_coords, bounds_info, centroid_coords): meta 
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
