# nre_0001_gen_detection

This project uses Google's Gemini Pro Vision model and Google Maps Static API to detect features (specifically irrigation pivots) in satellite imagery based on input GeoJSON polygons.

## Prerequisites

- Python 3.8+
- A Google Cloud Project with the following APIs enabled:
    - Google Gemini API
    - Google Maps Static API

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd nre_0001_gen_detection
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set up environment variables:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    - Edit `.env` and add your API keys:
        ```
        GOOGLE_API_KEY=your_gemini_api_key
        GOOGLE_MAPS_API_KEY=your_maps_api_key
        ```

## Usage

1.  Place your input GeoJSON file at `input/input.geojson`. The file should contain `Polygon` features representing the areas to scan.

2.  Run the detection script:
    ```bash
    python label_features.py
    ```

3.  The script will:
    - Read features from `input/input.geojson`.
    - Fetch a satellite image for each feature using the Google Maps Static API.
    - Send the image to the Gemini model to detect linear features (irrigation pivots).
    - Save the detected features as a new GeoJSON file at `output/output.geojson`.

## Output

The output file `output/output.geojson` will contain `LineString` features representing the detected structures. Each feature includes:
-   `input_feature_id`: The ID of the source polygon.
-   `confidence`: The model's confidence score (0.0 - 1.0).
-   `processed_at`: Timestamp of processing.
-   `imagery_source`: Source of the satellite imagery.
-   `source_metadata`: Details about the map image (zoom, center, size).
