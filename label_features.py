import os
import json
import time
import math
import requests
from datetime import datetime, timezone
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
INPUT_GEOJSON = "input/input.geojson"
OUTPUT_GEOJSON = "output/output.geojson"
FEATURE_PROMPT = "irrigation pivot"
MAP_SIZE = "640x640"
MAP_SCALE = 2 # High DPI
ZOOM_LEVEL = 16 # Adjust based on feature size

# ---------------------

def setup_gemini():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return None
    return genai.Client(api_key=GOOGLE_API_KEY)

def latlng_to_pixel(lat, lng, zoom):
    """Converts Lat/Lng to pixel coordinates at a given zoom level."""
    scale = 256 * 2**zoom
    x = (lng + 180) / 360 * scale
    sin_y = math.sin(lat * math.pi / 180)
    sin_y = min(max(sin_y, -0.9999), 0.9999)
    y = (0.5 - math.log((1 + sin_y) / (1 - sin_y)) / (4 * math.pi)) * scale
    return x, y

def pixel_to_latlng(px, py, zoom):
    """Converts pixel coordinates to Lat/Lng at a given zoom level."""
    scale = 256 * 2**zoom
    lng = (px / scale) * 360 - 180
    y = 0.5 - (py / scale)
    lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
    return lat, lng

def get_static_map_image(lat, lng, zoom, size, api_key):
    """Fetches a static map image from Google Maps API."""
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": size,
        "maptype": "satellite",
        "key": api_key,
        "scale": MAP_SCALE
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error fetching map: {response.status_code} - {response.text}")
        return None

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
        
        print(f"Token usage: {response.usage_metadata}")
        
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
                raise

        if isinstance(data, dict):
            data = [data]
            
        valid_data = []
        for item in data:
            if all(k in item for k in ['y1', 'x1', 'y2', 'x2', 'confidence']):
                valid_data.append(item)
            else:
                print(f"Skipping invalid item: {item}")
        
        return valid_data

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def process_feature(client, feature):
    """Processes a single GeoJSON feature."""
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    
    if geom.get("type") != "Polygon":
        print(f"Skipping non-polygon feature: {props.get('id')}")
        return []

    # Calculate centroid
    coords = geom["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    center_lon = sum(lons) / len(lons)
    center_lat = sum(lats) / len(lats)

    # Fetch map
    print(f"Fetching map for feature {props.get('id')} at {center_lat}, {center_lon}...")
    img = get_static_map_image(center_lat, center_lon, ZOOM_LEVEL, MAP_SIZE, GOOGLE_MAPS_API_KEY)
    
    if not img:
        return []

    # Detect features
    detections = get_feature_coordinates(client, img, FEATURE_PROMPT)
    if not detections:
        return []

    print(f"DEBUG: Raw detections: {detections}")

    output_features = []
    width, height = img.size # Should be 640x640 * scale (e.g. 1280x1280)
    
    # Calculate center pixel coordinates
    center_px, center_py = latlng_to_pixel(center_lat, center_lon, ZOOM_LEVEL)
    
    # Top-left pixel of the image in global pixel coordinates
    # Image is centered on center_lat, center_lon
    top_left_px = center_px - width / 2
    top_left_py = center_py - height / 2

    for d in detections:
        # Convert normalized coordinates to image pixels
        # Note: Gemini output is 0-1000. Image size might be different.
        # We need to map 0-1000 to 0-width/height
        
        img_x1 = d['x1'] / 1000 * width
        img_y1 = d['y1'] / 1000 * height
        img_x2 = d['x2'] / 1000 * width
        img_y2 = d['y2'] / 1000 * height
        
        # Convert image pixels to global pixels
        global_x1 = top_left_px + img_x1
        global_y1 = top_left_py + img_y1
        global_x2 = top_left_px + img_x2
        global_y2 = top_left_py + img_y2
        
        # Convert global pixels to Lat/Lng
        lat1, lon1 = pixel_to_latlng(global_x1, global_y1, ZOOM_LEVEL)
        lat2, lon2 = pixel_to_latlng(global_x2, global_y2, ZOOM_LEVEL)
        
        # Create GeoJSON feature
        out_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon1, lat1], [lon2, lat2]]
            },
            "properties": {
                "input_feature_id": props.get("id"),
                "confidence": d['confidence'],
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "imagery_source": "Google Maps Static API",
                "source_metadata": {
                    "zoom": ZOOM_LEVEL,
                    "center": [center_lat, center_lon],
                    "size": MAP_SIZE,
                    "scale": MAP_SCALE
                }
            }
        }
        output_features.append(out_feature)
        
    return output_features

def main():
    if not GOOGLE_MAPS_API_KEY:
        print("Error: GOOGLE_MAPS_API_KEY not found in environment variables.")
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
    
    for feature in input_data.get("features", []):
        detected_features = process_feature(client, feature)
        all_output_features.extend(detected_features)
        time.sleep(1) # Rate limiting

    output_collection = {
        "type": "FeatureCollection",
        "features": all_output_features
    }

    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump(output_collection, f, indent=2)
    
    print(f"Saved {len(all_output_features)} detected features to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
