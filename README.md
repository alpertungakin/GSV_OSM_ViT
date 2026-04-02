## `floors.py`

This script estimates the number of floors in a building by counting rows of windows. 

**Functionality:**
1. **Input:** Prompts the user for a Google Maps API Key and opens a file dialog to select a shapefile containing building coordinates.
2. **Image Retrieval:** Finds the nearest Street View panorama for each building, downloads the image tiles, and stitches them into a full panorama.
3. **Isolation:** Calculates the building's position relative to the camera and uses a pre-trained Mask R-CNN model to segment the specific building facade from the background.
4. **Correction:** Extracts the building crop and applies an orthorectification process to flatten the perspective.
5. **Detection:** Uses a SAM 3 model to identify windows and doors, utilizing negative prompts to ignore objects like cars and trees.
6. **Counting:** Groups the detected windows by their vertical positions using the DBSCAN algorithm. The number of vertical groups serves as the floor count estimate.
7. **Output:** Generates annotated images of the facades with horizontal lines denoting the floors, and compiles the final floor counts into an Excel file (`building_floor_counts.xlsx`).

## `amenity.py`

This script attempts to classify the primary use of a building and identify potential amenities to verify OpenStreetMap (OSM) tags.

**Functionality:**
1. **Input:** Similar to the first script, it takes a Google Maps API Key and a shapefile. The shapefile is expected to contain existing `building` and `amenity` attributes.
2. **Image Retrieval & Isolation:** Reuses the pipeline from `floors.py` to download panoramas, stitch them, and isolate the target building facade using Mask R-CNN.
3. **Classification:** Passes the cropped facade image to a vision-language model (SigLIP). 
4. **Categorization:** The model calculates confidence scores against two predefined sets of text prompts:
    * Building types (e.g., residential, commercial, industrial, church).
    * Amenity types (e.g., cafe, pharmacy, school, none).
5. **Output:** Saves cropped images of the facades and generates an Excel file (`siglip_osm_classifications.xlsx`) that aligns the original OSM tags with the SigLIP model's predictions and confidence percentages.

## Requirements

Both scripts require a standard data science environment along with specific geospatial and machine learning libraries.

**Key Dependencies:**
* `torch`, `torchvision`, `transformers`
* `opencv-python`, `Pillow`, `scikit-image`, `scikit-learn`
* `geopandas`, `pandas`, `pyproj`
* `sam3`
* `equilib`

**Required Model Weights:**
The scripts expect the following local weight files in a `weights/` directory:
* `weights/panorama_maskrcnn_model.pth`
* `weights/sam3.pt`
*(Note: The SigLIP model in `amenity.py` is downloaded automatically via the `transformers` library).*

## Usage

1. Run either script from the command line:
   ```bash
   python floors.py
   # or
   python amenity.py
   ```
2. Enter your Google Maps API key when prompted in the terminal.
3. Select your `.shp` file in the popup dialog window.
4. The scripts will create a timestamped session directory to store all downloaded images, intermediate crops, and the final Excel reports.