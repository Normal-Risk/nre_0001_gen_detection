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
import base64
from google.genai import types
from google import genai as genai_client

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
    """Initializes GEE."""
    # 1. Initialize Gemini (API key is used directly by genai_client later)
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=GOOGLE_API_KEY) # Still configure for potential direct calls or other uses

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

PROMPT_TEXT = (
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

def image_to_base64(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

def prepare_batch_request(feature, i, client):
    """Prepares a single batch request entry."""
    feature_id = feature.get('id', f'feature_{i}')
    geom = feature['geometry']
    if geom['type'] != 'Polygon': return None
    
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
        
    except Exception as e:
        print(f"Error calculating geometry for {feature_id}: {e}")
        return None
    
    # Get Metadata (All available)
    metadata_list = get_gee_metadata(coords[0], feature_id) 
    
    if not metadata_list:
        print(f"  No images found for {feature_id}")
        return None

    # Download Images
    parts = [{"text": PROMPT_TEXT}]
    valid_metas = []
    
    for meta in metadata_list:
        img = download_image(meta, region_coords)
        if img:
            b64_data = image_to_base64(img)
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_data
                }
            })
            valid_metas.append(meta)
            
    if not valid_metas:
        return None

    # Construct Request Object
    request_entry = {
        "key": feature_id,
        "request": {
            "contents": [{"parts": parts}],
            "generation_config": {
                "response_mime_type": "application/json"
            }
        },
        # Store metadata in a separate file or map to use during post-processing
        # Since the batch API only returns the model response, we need to save the context (bounds, dates) locally.
        "custom_context": {
            "bounds_info": bounds_info,
            "valid_metas": valid_metas,
            "feature_id": feature_id
        }
    }
    return request_entry

def process_batch_result(response_text, context):
    """Processes the model output using the saved context."""
    try:
        # Clean markdown
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        data = json.loads(text)
        if not isinstance(data, list) or not data:
            return []
            
        d = data[0] # Consensus object
        
        min_lon, max_lon, min_lat, max_lat = context['bounds_info']
        valid_metas = context['valid_metas']
        
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
        return [feature]
        
    except Exception as e:
        print(f"Error processing result for {context['feature_id']}: {e}")
        return []

def main():
    initialize_services()
    
    # Load Input
    try:
        with open(INPUT_GEOJSON, 'r') as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_GEOJSON} not found.")
        return
    
    print(f"Preparing batch requests for {len(input_data['features'])} features...")
    
    batch_requests = []
    context_map = {} # Store context by key to retrieve later
    
    # We can still use threads to download images faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(prepare_batch_request, feature, i, None)
            for i, feature in enumerate(input_data['features'])
        ]
        
        for future in concurrent.futures.as_completed(futures):
            entry = future.result()
            if entry:
                # Separate the API request from our local context
                req_data = {
                    "key": entry["key"],
                    "request": entry["request"]
                }
                batch_requests.append(req_data)
                context_map[entry["key"]] = entry["custom_context"]
                print(f"  Prepared request for {entry['key']}")

    if not batch_requests:
        print("No valid requests generated.")
        return

    # Write JSONL
    jsonl_filename = "batch_input.jsonl"
    with open(jsonl_filename, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
            
    print(f"Created {jsonl_filename} with {len(batch_requests)} requests.")

    # Initialize GenAI Client for Batch
    # Note: The 'genai' module used earlier is google.generativeai
    # The Batch API is better supported in the newer google-genai SDK or via REST, 
    # but let's try to use the google.generativeai client if it supports it, 
    # otherwise we might need to use the 'google-genai' package if installed.
    # Based on docs, we should use `google.genai.Client`.
    
    try:
        client = genai_client.Client(api_key=os.environ["GOOGLE_API_KEY"])
    except Exception as e:
        print(f"Error initializing GenAI Client: {e}")
        return

    # Upload File
    print("Uploading batch file...")
    batch_file = client.files.upload(
        file=jsonl_filename,
        config=types.UploadFileConfig(mime_type='application/json')
    )
    print(f"File uploaded: {batch_file.name}")

    # Create Batch Job
    print("Submitting batch job...")
    job = client.batches.create(
        model="gemini-3-pro-preview", # Using 1.5 Pro 002 for batch support
        src=batch_file.name,
        config=types.CreateBatchJobConfig(
            display_name="pivot_detection_batch"
        )
    )
    print(f"Job created: {job.name}")
    print(f"View job: https://aistudio.google.com/app/batch/{job.name}")

    # Poll for Completion
    print("Waiting for job to complete...")
    while True:
        job = client.batches.get(name=job.name)
        print(f"  Status: {job.state}")
        
        if job.state == "JOB_STATE_SUCCEEDED":
            break
        elif job.state == "JOB_STATE_FAILED" or job.state == "JOB_STATE_CANCELLED":
            print(f"Job failed: {job.error}")
            return
            
        time.sleep(30)

    # Download Results
    print("Job succeeded! Downloading results...")
    output_file_name = job.output_file.name
    content = client.files.content(name=output_file_name)
    
    # Process Results
    results_jsonl = content.decode('utf-8')
    final_features = []
    
    for line in results_jsonl.splitlines():
        if not line.strip(): continue
        res = json.loads(line)
        key = res['custom_id'] # The API returns 'custom_id' matching our 'key'
        
        # Extract the response text
        # The structure depends on the API version, but typically:
        # response -> candidates -> content -> parts -> text
        try:
            response_obj = res['response']['body']
            # It might be a nested JSON string or object depending on SDK
            if isinstance(response_obj, str):
                response_obj = json.loads(response_obj)
                
            # Navigate to text
            # This path might need adjustment based on exact API response structure
            # But let's assume standard GenerateContentResponse structure
            candidates = response_obj.get('candidates', [])
            if candidates:
                parts = candidates[0].get('content', {}).get('parts', [])
                if parts:
                    text_response = parts[0].get('text', '')
                    
                    if key in context_map:
                        features = process_batch_result(text_response, context_map[key])
                        final_features.extend(features)
                        print(f"  Processed result for {key}")
        except Exception as e:
            print(f"Error parsing response for {key}: {e}")

    # Save Output
    with open(OUTPUT_GEOJSON, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": final_features}, f, indent=2)
        
    print(f"Saved {len(final_features)} features to {OUTPUT_GEOJSON}")

if __name__ == "__main__":
    main()
