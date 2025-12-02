# nre_0001_gen_detection

This project uses Google's Gemini Pro Vision model and **Google Earth Engine (GEE)** to detect features (specifically irrigation pivots) in satellite imagery based on input GeoJSON polygons.

It fetches imagery from the **USDA NAIP (National Agriculture Imagery Program)** dataset, processing all available historical images for each location to maximize detection opportunities.

## Prerequisites

- Python 3.8+
- A Google Cloud Project with the following APIs enabled:
    - Google Gemini API
    - Google Earth Engine API
- **Google Earth Engine Access**: You must be registered for Earth Engine.

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

3.  **Authentication**:
    You have two options for authenticating with Google Earth Engine:

    *   **Option A: Service Account (Recommended)**
        1.  Create a Service Account in Google Cloud.
        2.  Grant it the "Earth Engine Resource Viewer" role.
        3.  Create and download a JSON key file.
        4.  Set the path in your `.env` file (see below).

    *   **Option B: Interactive Login**
        1.  Run `earthengine authenticate` in your terminal and follow the instructions.

4.  Set up environment variables:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    - Edit `.env` and add your keys:
        ```
        GOOGLE_API_KEY=your_gemini_api_key
        GEE_SERVICE_ACCOUNT_KEY_FILE=/absolute/path/to/your/service-account-key.json
        ```
        *(Leave `GEE_SERVICE_ACCOUNT_KEY_FILE` empty if using interactive login)*

## Usage

1.  Place your input GeoJSON file at `input/input.geojson`. The file should contain `Polygon` features representing the areas to scan.

2.  Run the detection script:
    ```bash
    python label_features.py
    ```

3.  The script will:
    - Read features from `input/input.geojson`.
    - Query Google Earth Engine for all available NAIP images intersecting each feature.
    - Download a high-resolution thumbnail for each image date.
    - Send the image to the Gemini model to detect linear features (irrigation pivots).
    - Save the detected features as a new GeoJSON file at `output/output.geojson`.

## Output

The output file `output/output.geojson` will contain `LineString` features representing the detected structures. Each feature includes:
-   `input_feature_id`: The ID of the source polygon.
-   `confidence`: The model's confidence score (0.0 - 1.0).
-   `processed_at`: Timestamp of processing.
-   `imagery_source`: "Google Earth Engine (USDA/NAIP/DOQQ)".
-   `source_metadata`:
    -   `gee_id`: The unique ID of the image in Earth Engine.
    -   `date`: The acquisition date of the image.
    -   `collection`: The source collection.
