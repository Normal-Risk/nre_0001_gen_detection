# nre_0001_gen_detection

This project uses Google's Gemini Pro Vision model and **Google Earth Engine (GEE)** to detect features (specifically irrigation pivots) in satellite imagery based on input GeoJSON polygons.

It fetches imagery from the **USDA NAIP (National Agriculture Imagery Program)** dataset, processing all available historical images for each location to maximize detection opportunities.

## Prerequisites

- Python 3.9+
- A Google Cloud Project with the following APIs enabled:
    - Google Gemini API
    - Google Earth Engine API
- **Google Earth Engine Access**: You must be registered for Earth Engine.
- **Python Libraries**: `shapely`, `pyproj` (in addition to standard requirements).

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

### 1. Preprocessing (Filtering)

Before running the detection, you can filter your raw input data (e.g., from `~/Downloads/irregular_circles.geojson`) to select specific features (e.g., circles in Kansas with area 0.4-0.6 sq km).

Run the filter script:
```bash
python filter.py
```
This will create `input/filtered_kansas_circles.geojson`.

### 2. Feature Detection (Batch Processing)

Run the main detection script, which processes features in chunks to manage limits and ensure reliability:

```bash
python label_areas_advanced_batched.py
```

**How it works:**
- Reads features from `input/filtered_kansas_circles.geojson`.
- Splits the data into **chunks of 1000 features**.
- For each chunk:
    - Creates a Gemini Batch Job (e.g., `pivot_detection_batch_0`).
    - Downloads NAIP imagery from Google Earth Engine.
    - Sends images to Gemini to detect pivot centerpoints and arms.
    - Saves the results to a separate output file (e.g., `output/centerpoints_part_0.geojson`).

## Output

The script generates multiple output files in the `output/` directory, named `centerpoints_part_N.geojson`.

Each file contains a `FeatureCollection` of points representing the detected pivot centers. Properties include:
-   `confidence`: Model confidence score (0.0 - 1.0).
-   `field_coverage`: Estimated irrigated sector angles (e.g., `[0, 270]`).
-   `avg_length_meters`: Average length of the pivot arm.
-   `arms`: A list of detected arms for each image date, including:
    -   `image_date`: Date of the satellite image.
    -   `geometry`: `LineString` representing the arm.
    -   `length_meters`: Length of the arm.
-   `imagery_source`: "Google Earth Engine (USDA/NAIP/DOQQ)".
