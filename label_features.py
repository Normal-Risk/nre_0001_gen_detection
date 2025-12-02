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

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEE_SERVICE_ACCOUNT_KEY_FILE = os.getenv("GEE_SERVICE_ACCOUNT_KEY_FILE")
INPUT_GEOJSON = "input/input.geojson"
OUTPUT_GEOJSON = "output/output.geojson"
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
            credentials = service_account.Credentials.from_service_account_file(GEE_SERVICE_ACCOUNT_KEY_FILE)
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

def get_gee_images(geometry_coords, feature_id):
    """
    Fetches all available images for a given geometry from the NAIP collection.
    Returns a list of tuples: (PIL.Image, metadata_dict)
    """
    try:
        # Create an ee.Geometry from the input coordinates
        # GeoJSON coordinates are [lon, lat], which GEE expects
        roi = ee.Geometry.Polygon(geometry_coords)
        
        # Buffer the region slightly to ensure we get the whole feature plus context
        # NAIP images are tiles, so we might need to mosaic if the feature crosses tiles.
        # However, for simplicity, we'll filter by bounds and mosaic the overlapping images for each distinct date.
        # Actually, NAIP is a collection of images. We want to see the feature at different times.
        # So we should group by date.
        
        collection = ee.ImageCollection(GEE_COLLECTION)\
            .filterBounds(roi)\
            .sort('system:time_start', False) # Newest first

        # Get the list of images (client-side)
        # Limit to a reasonable number to avoid timeouts if there are tons
        image_list = collection.toList(50) 
        count = image_list.size().getInfo()
        
        print(f"Found {count} images for feature {feature_id}")
        
        results = []
        
        # Iterate through the images
        for i in range(count):
            ee_img = ee.Image(image_list.get(i))
            info = ee_img.getInfo()
            props = info.get('properties', {})
            img_id = info.get('id')
            timestamp = props.get('system:time_start')
            
            # Convert timestamp to readable date
            date_str = "Unknown"
            if timestamp:
                dt = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
                date_str = dt.isoformat()
            
            print(f"  Processing image {i+1}/{count}: {img_id} ({date_str})")
            
            # Select RGB bands (NAIP has R, G, B, N)
            vis_params = {
                'min': 0,
                'max': 255,
                'bands': ['R', 'G', 'B'],
                'dimensions': 1024, # Max dimension
                'region': roi.buffer(REGION_BUFFER).bounds().getInfo()['coordinates'] # Use buffered bounds
            }
            
            try:
                url = ee_img.getThumbURL(vis_params)
                response = requests.get(url)
                if response.status_code == 200:
                    img_data = Image.open(BytesIO(response.content))
                    
                    metadata = {
                        "gee_id": img_id,
                        "date": date_str,
                        "timestamp": timestamp,
                        "collection": GEE_COLLECTION,
                        "scale": SCALE
                    }
                    results.append((img_data, metadata))
                else:
                    print(f"    Failed to download thumbnail: {response.status_code}")
            except Exception as e:
                print(f"    Error generating thumbnail: {e}")
                
        return results

    except Exception as e:
        print(f"Error fetching GEE images: {e}")
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
        
        # print(f"Token usage: {response.usage_metadata}")
        
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

def process_feature(client, feature):
    """Processes a single GeoJSON feature against all available GEE images."""
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    feature_id = props.get("id", "unknown")
    
    if geom.get("type") != "Polygon":
        print(f"Skipping non-polygon feature: {feature_id}")
        return []

    # Get coordinates for GEE
    coords = geom["coordinates"] # GEE expects [[[lon, lat], ...]] for Polygon

    # Fetch images from GEE
    images_and_metadata = get_gee_images(coords, feature_id)
    
    if not images_and_metadata:
        print(f"No images found for feature {feature_id}")
        return []

    all_detections_for_feature = []

    for img, metadata in images_and_metadata:
        # Detect features
        detections = get_feature_coordinates(client, img, FEATURE_PROMPT)
        if not detections:
            continue

        width, height = img.size
        
        # For mapping back to lat/lng, we need the bounds of the image we requested.
        # In get_gee_images, we requested the bounds of the ROI + buffer.
        # We can re-calculate the bounds here or pass them back.
        # For simplicity, let's assume the image corresponds exactly to the requested region.
        # But wait, getThumbURL might fit the region into the dimensions.
        # A more robust way is to ask GEE for the bounds of the resulting image, but that's complex.
        # Approximation: We know the ROI.
        
        # Let's recalculate the bounds used in get_gee_images
        roi = ee.Geometry.Polygon(coords)
        buffered_roi = roi.buffer(REGION_BUFFER).bounds().getInfo()
        roi_coords = buffered_roi['coordinates'][0] # List of [lon, lat]
        
        # Find min/max lon/lat
        lons = [c[0] for c in roi_coords]
        lats = [c[1] for c in roi_coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        # Image covers [min_lon, min_lat] to [max_lon, max_lat]
        # width corresponds to (max_lon - min_lon)
        # height corresponds to (max_lat - min_lat)
        
        for d in detections:
            # Normalized 0-1000
            norm_x1 = d['x1'] / 1000
            norm_y1 = d['y1'] / 1000
            norm_x2 = d['x2'] / 1000
            norm_y2 = d['y2'] / 1000
            
            # Map to Lat/Lng
            # x is longitude (left to right)
            # y is latitude (top to bottom? No, map coordinates: y increases upwards (North))
            # BUT, in images, y=0 is top. So y increases downwards.
            # Latitude: min_lat is bottom, max_lat is top.
            # So y=0 (top) -> max_lat. y=1 (bottom) -> min_lat.
            
            lon1 = min_lon + norm_x1 * (max_lon - min_lon)
            lat1 = max_lat - norm_y1 * (max_lat - min_lat)
            lon2 = min_lon + norm_x2 * (max_lon - min_lon)
            lat2 = max_lat - norm_y2 * (max_lat - min_lat)
            
            # Create GeoJSON feature
            out_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]]
                },
                "properties": {
                    "input_feature_id": feature_id,
                    "confidence": d['confidence'],
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "imagery_source": "Google Earth Engine (USDA/NAIP/DOQQ)",
                    "source_metadata": metadata
                }
            }
            all_detections_for_feature.append(out_feature)
            
    return all_detections_for_feature

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

    all_output_features = []
    
    features = input_data.get("features", [])
    print(f"Processing {len(features)} input features...")
    
    for feature in features:
        detected_features = process_feature(client, feature)
        all_output_features.extend(detected_features)
        # time.sleep(1) # Rate limiting not strictly needed for GEE/Gemini in this loop structure but good practice

    output_collection = {
        "type": "FeatureCollection",
        "features": all_output_features
    }

    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump(output_collection, f, indent=2)
    
    print(f"Saved {len(all_output_features)} detected features to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
