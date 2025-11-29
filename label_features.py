import os
import glob
import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# The feature you want to detect. CHANGE THIS.
FEATURE_PROMPT = "irrigation pivot" 

# ---------------------

def setup_gemini():
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        return None
    return genai.Client(api_key=GOOGLE_API_KEY)

def get_feature_coordinates(client, image_path, feature_name):
    """
    Sends image to Gemini and asks for line coordinates.
    Returns a list of [y1, x1, y2, x2] normalized to 0-1000.
    """
    print(f"Processing {image_path}...")
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    prompt = (
        f"Analyze this satellite image and identify the {feature_name}. "
        "It appears as a linear structure. Draw a line segment along the length of the pivot arm. "
        "Return the start and end coordinates of this line segment. "
        "Output the result as a JSON object with keys 'y1', 'x1', 'y2', 'x2', "
        "where values are integers normalized to the range [0, 1000]."
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
        elif "```" in text: # Handle case where json tag is missing
             start = text.find("```") + 3
             end = text.find("```", start)
             if end != -1:
                text = text[start:end]

        # Clean up whitespace
        text = text.strip()
        
        # Fix single quotes if present
        text = text.replace("'", '"')
        
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: try to find first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                text = text[start:end+1]
                data = json.loads(text)
            else:
                raise

        # Validate keys
        if all(k in data for k in ['y1', 'x1', 'y2', 'x2']):
            return data
        else:
            print(f"Unexpected JSON structure: {data}")
            return None

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def draw_label(image_path, coords, output_path):
    """
    Draws a line on the image and saves it.
    """
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            width, height = img.size
            
            # Convert normalized 0-1000 coordinates to pixels
            y1 = coords['y1'] / 1000 * height
            x1 = coords['x1'] / 1000 * width
            y2 = coords['y2'] / 1000 * height
            x2 = coords['x2'] / 1000 * width
            
            # Draw line
            # Color red, width 5
            draw.line([x1, y1, x2, y2], fill="red", width=5)
            
            img.save(output_path)
            print(f"Saved labeled image to {output_path}")
            
    except Exception as e:
        print(f"Error drawing/saving image: {e}")

def main():
    client = setup_gemini()
    if not client:
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all images in input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
        # Also check uppercase
        image_files.extend(glob.glob(os.path.join(INPUT_DIR, ext.upper())))

    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images.")

    for img_file in image_files:
        filename = os.path.basename(img_file)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        coords = get_feature_coordinates(client, img_file, FEATURE_PROMPT)
        
        if coords:
            draw_label(img_file, coords, output_path)
        else:
            print(f"Could not detect feature in {filename}")
        
        # Sleep briefly to avoid rate limits if processing many
        time.sleep(1)

if __name__ == "__main__":
    main()
