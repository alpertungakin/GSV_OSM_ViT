import os
# FIX FOR OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import math
import cv2
import torch
import requests
import pyproj
import numpy as np
import geopandas as gp
import pandas as pd
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import streetview
from datetime import datetime
from transformers import AutoProcessor, AutoModel
from equilib import Equi2Pers

geodesic = pyproj.Geod(ellps='WGS84')

def get_bearing(pano_lat, pano_lon, bldg_lat, bldg_lon):
    fwd_azimuth, back_azimuth, distance = geodesic.inv(pano_lon, pano_lat, bldg_lon, bldg_lat)
    if fwd_azimuth < 0:
        fwd_azimuth += 360
    return fwd_azimuth

def get_target_x_pixel(bearing, pano_heading, img_width):
    relative_angle = (bearing - pano_heading + 180) % 360 - 180
    x_pixel = (relative_angle / 360.0 + 0.5) * img_width
    return int(x_pixel)

def get_pano_metadata(pano_id, api_key):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?pano={pano_id}&key={api_key}"
    resp = requests.get(url).json()
    if resp.get('status') == 'OK':
        return {
            'lat': resp['location']['lat'],
            'lng': resp['location']['lng'],
            'heading': resp.get('copyright', 0) if isinstance(resp.get('copyright'), (int, float)) else 0 
        }
    return None


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def extract_facade(equi_img, bbox, mask_center_x=None):
    H, W, _ = equi_img.shape
    min_col, min_row, max_col, max_row = bbox

    if mask_center_x is not None:
        center_x = mask_center_x
    else:
        center_x = min_col + (max_col - min_col) / 2.0
        
    yaw = (0.5 - center_x / W) * 2 * np.pi
    pitch = 0.0

    box_width = max(max_col - min_col, 1)
    fov_x_deg = min((box_width / W) * 360.0, 160.0)

    angle_top = (0.5 - min_row / H) * 180.0
    angle_bottom = (0.5 - max_row / H) * 180.0
    max_y_angle = max(abs(angle_top), abs(angle_bottom))
    fov_y_deg = min(max_y_angle * 2.0 * 1.1, 160.0)

    out_width = max(int(box_width * 1.5), 10) 
    tan_y = np.tan(np.radians(max(fov_y_deg, 1) / 2.0))
    tan_x = np.tan(np.radians(max(fov_x_deg, 1) / 2.0))
    out_height = max(int(out_width * (tan_y / tan_x)), 10) 
    
    MAX_DIM = 2048
    if out_width > MAX_DIM or out_height > MAX_DIM:
        scale = MAX_DIM / max(out_width, out_height)
        out_width = max(int(out_width * scale), 10)
        out_height = max(int(out_height * scale), 10)

    equi2pers = Equi2Pers(height=out_height, width=out_width, fov_x=fov_x_deg, mode='bilinear')
    equi_img_tensor = np.transpose(equi_img, (2, 0, 1))
    rots = {'roll': 0.0, 'pitch': pitch, 'yaw': yaw}
    
    pers_img_tensor = equi2pers(equi=equi_img_tensor, rots=rots)
    return np.transpose(pers_img_tensor, (1, 2, 0)).astype(np.uint8)

def classify_osm_tags_siglip(image_path, model, processor, device):
    image = Image.open(image_path).convert('RGB')

    building_tags = {
        "residential": "a photo of a residential building, house, or apartment block exterior",
        "commercial": "a photo of a commercial office building facade",
        "retail": "a photo of a retail building or shopping center facade",
        "industrial": "a photo of an industrial building or warehouse exterior",
        "civic": "a photo of a civic or government public building facade",
        "mosque": "a photo of a mosque exterior with Islamic architectural features",
        "church": "a photo of a church exterior with Christian architectural features",
    }

    amenity_tags = {
        "cafe": "a photo of a cafe or coffee shop storefront",
        "pharmacy": "a photo of a pharmacy, drugstore or chemist facade",
        "car wash": "a photo of a car wash facility exterior",
        "fast_food": "a photo of a fast food restaurant facade",
        "restaurant": "a photo of a restaurant exterior",
        "school": "a photo of a school with student activity visible",
        "college": "a photo of a college or university building facade",
        "none": "a photo of a plain building facade with no storefront or public amenity"
    }
    
    b_keys = list(building_tags.keys())
    a_keys = list(amenity_tags.keys())
    all_texts = list(building_tags.values()) + list(amenity_tags.values())

    inputs = processor(
        text=all_texts, 
        images=image, 
        padding="max_length", 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = torch.sigmoid(logits_per_image).squeeze()

    b_probs = probs[:len(b_keys)]
    a_probs = probs[len(b_keys):]

    best_b_idx = b_probs.argmax().item()
    best_b_tag = b_keys[best_b_idx]
    best_b_score = b_probs[best_b_idx].item()

    best_a_idx = a_probs.argmax().item()
    best_a_tag = a_keys[best_a_idx]
    best_a_score = a_probs[best_a_idx].item()

    return (best_b_tag, best_b_score), (best_a_tag, best_a_score)


def main():
    GOOGLE_MAPS_API_KEY = input("Enter your Google Maps API Key: ").strip()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # --- UPDATED: New Session Directory Naming ---
    session_dir = datetime.now().strftime("session_Siglip_%Y%m%d_%H%M%S")
    os.makedirs(session_dir, exist_ok=True)
    print(f"Created output directory: {session_dir}")

    # 1. Select Shapefile via Dialog
    root = tk.Tk()
    root.withdraw()
    shp_path = filedialog.askopenfilename(
        title="Select Building Shapefile",
        filetypes=[("Shapefiles", "*.shp"), ("All files", "*.*")]
    )
    if not shp_path:
        print("No shapefile selected. Exiting.")
        return

    print(f"Loading shapefile: {shp_path}")
    buildings = gp.read_file(shp_path)

    # 2. Load Models
    print("Loading Mask R-CNN model for building extraction...")
    mask_model = get_model_instance_segmentation(num_classes=2)
    # Ensure this path points to your weights
    mask_model.load_state_dict(torch.load("weights/panorama_maskrcnn_model.pth", map_location=device))
    mask_model.to(device)
    mask_model.eval()

    print("Loading SigLIP model for OSM tag verification...")
    siglip_id = "google/siglip-base-patch16-224"
    siglip_processor = AutoProcessor.from_pretrained(siglip_id)
    siglip_model = AutoModel.from_pretrained(siglip_id).to(device)
    siglip_model.eval()

    # 3. Authenticate Google API
    headers = {'Content-Type': 'application/json'}
    data = """{"mapType": "streetview","language": "en-US", "region": "US"}"""
    response = requests.post(f"https://tile.googleapis.com/v1/createSession?key={GOOGLE_MAPS_API_KEY}", headers=headers, data=data)
    SESSION_TOKEN = response.json()['session']
    
    classification_results = []
    
    # 4. Process Each Building
    for idx, row in buildings.iterrows():
        osm_id = str(row.get('osm_id', f'unknown_{idx}'))
        
        osm_bldg_tag = row.get('building', 'N/A')
        osm_amenity_tag = row.get('amenity', 'N/A')

        bldg_lon, bldg_lat = row.geometry.centroid.x, row.geometry.centroid.y
        print(f"\n--- Processing Shapefile Building {osm_id} ---")
        
        result_row = {
            "Building_ID": f"{osm_id}_UNKNOWN",
            "OSM_Building": osm_bldg_tag,
            "OSM_Amenity": osm_amenity_tag,
            "SigLIP_Building": "Failed/No Pano",
            "SigLIP_B_Conf": 0.0,
            "SigLIP_Amenity": "Failed/No Pano",
            "SigLIP_A_Conf": 0.0
        }

        panos = streetview.panoids(lat=bldg_lat, lon=bldg_lon)
        if not panos: 
            classification_results.append(result_row)
            continue
        
        try:
            panos_df = pd.DataFrame(panos)
            pano_id = panos_df.sort_values(by='year', ascending=False).iloc[0]["panoid"]
        except KeyError:
            pano_id = panos[0]["panoid"]
        
        result_row["Building_ID"] = f"{osm_id}_{pano_id}"

        meta = get_pano_metadata(pano_id, GOOGLE_MAPS_API_KEY)
        if not meta: 
            classification_results.append(result_row)
            continue
            
        bearing = get_bearing(meta['lat'], meta['lng'], bldg_lat, bldg_lon)
        pano_heading = meta['heading'] 

        # --- Downloading & Stitching ---
        print(f"  Downloading tiles for pano: {pano_id}...")
        
        urls = [
            f"https://tile.googleapis.com/v1/streetview/tiles/2/0/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/1/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/2/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/3/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/0/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/1/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/2/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}",
            f"https://tile.googleapis.com/v1/streetview/tiles/2/3/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}"
        ]

        # --- Downloading & Stitching ---
        print(f"  Downloading tiles for pano: {pano_id}...")
        
        download_success = True
        for i, url in enumerate(urls):
            response = requests.get(url)
            # Check if the download was actually successful
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                with open(f"{session_dir}/{pano_id}_{i}.jpg", 'wb') as f:
                    f.write(response.content)
            else:
                print(f"  -> Error downloading tile {i}. Google API returned status: {response.status_code}")
                download_success = False
                break # Stop downloading the rest of the tiles for this pano

        # If any tile failed, skip stitching and move to the next building
        if not download_success:
            print(f"  -> Skipping building {osm_id} due to missing panorama tiles.")
            result_row["SigLIP_Building"] = "Pano Download Failed"
            result_row["SigLIP_Amenity"] = "Pano Download Failed"
            classification_results.append(result_row)
            continue

        images_r1 = [Image.open(f"{session_dir}/{pano_id}_{i}.jpg") for i in range(4)]
        images_r2 = [Image.open(f"{session_dir}/{pano_id}_{i}.jpg") for i in range(4, 8)]
        
        widths, heights = zip(*(i.size for i in images_r1))
        total_width, max_height = sum(widths), max(heights)
        
        new_im_r1 = Image.new('RGB', (total_width, max_height))
        new_im_r2 = Image.new('RGB', (total_width, max_height))
        new_im = Image.new('RGB', (total_width, 2*max_height))

        x_offset1, x_offset2, y_offset = 0, 0, 0

        for im in images_r1:
            new_im_r1.paste(im, (x_offset1,0))
            x_offset1 += im.size[0]

        for im in images_r2:
            new_im_r2.paste(im, (x_offset2,0))
            x_offset2 += im.size[0]

        for img in [new_im_r1, new_im_r2]:
            new_im.paste(img, (0, y_offset))
            y_offset += max_height

        stitched_path = f"{session_dir}/{pano_id}.jpg"
        new_im.save(stitched_path)
        
        for i in range(8):
            os.remove(f"{session_dir}/{pano_id}_{i}.jpg")
        
        orig_img_cv = cv2.imread(stitched_path)
        if orig_img_cv is None: 
            classification_results.append(result_row)
            continue
            
        orig_img_rgb = cv2.cvtColor(orig_img_cv, cv2.COLOR_BGR2RGB)
        orig_H, orig_W, _ = orig_img_rgb.shape

        target_x = get_target_x_pixel(bearing, pano_heading, orig_W)

        INFER_W, INFER_H = 1024, 512
        infer_img = cv2.resize(orig_img_rgb, (INFER_W, INFER_H))
        img_tensor = F.to_tensor(infer_img).to(device)

        with torch.no_grad():
            prediction = mask_model([img_tensor])[0]

        scores = prediction['scores'].cpu().numpy()
        valid_indices = scores >= 0.5
        valid_boxes, valid_masks = prediction['boxes'].cpu().numpy()[valid_indices], prediction['masks'].cpu().numpy()[valid_indices]
        scale_x, scale_y = orig_W / INFER_W, orig_H / INFER_H

        matched_box_index, closest_distance = -1, float('inf')
        for i, box in enumerate(valid_boxes):
            x_min, x_max = box[0] * scale_x, box[2] * scale_x
            if x_min <= target_x <= x_max:
                box_center_x = (x_min + x_max) / 2.0
                dist = abs(target_x - box_center_x)
                if dist < closest_distance:
                    closest_distance, matched_box_index = dist, i

        result_row["SigLIP_Building"] = "AI Segment Failed"
        result_row["SigLIP_Amenity"] = "AI Segment Failed"

        if matched_box_index != -1:
            box = valid_boxes[matched_box_index]
            mask = valid_masks[matched_box_index, 0]
            scaled_box = [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y]
            binary_mask = (mask > 0.5).astype(np.uint8)
            M = cv2.moments(binary_mask)
            true_center_x = (M["m10"] / M["m00"]) * scale_x if M["m00"] != 0 else None

            try:
                facade_img = extract_facade(orig_img_rgb, scaled_box, mask_center_x=true_center_x)
            except Exception as e:
                print(f"  -> Skipping building {osm_id} due to geometry extraction error: {e}")
                classification_results.append(result_row)
                continue

            # Save the cropped facade for SigLIP (and manual review later if needed)
            temp_facade_name = f"{session_dir}/cropped_facade_{osm_id}_{pano_id}.jpg"
            cv2.imwrite(temp_facade_name, cv2.cvtColor(facade_img, cv2.COLOR_RGB2BGR))
            
            # --- SigLIP Classification ---
            print(f"  -> Classifying OSM attributes via SigLIP...")
            try:
                (sig_b_tag, sig_b_conf), (sig_a_tag, sig_a_conf) = classify_osm_tags_siglip(
                    temp_facade_name, siglip_model, siglip_processor, device
                )
                result_row["SigLIP_Building"] = sig_b_tag
                result_row["SigLIP_B_Conf"] = round(sig_b_conf * 100, 1)
                result_row["SigLIP_Amenity"] = sig_a_tag
                result_row["SigLIP_A_Conf"] = round(sig_a_conf * 100, 1)
                print(f"     * Predicted: {sig_b_tag} ({result_row['SigLIP_B_Conf']}%) | {sig_a_tag} ({result_row['SigLIP_A_Conf']}%)")
            except Exception as e:
                print(f"  -> SigLIP Classification failed: {e}")
                result_row["SigLIP_Building"] = "Error"
                result_row["SigLIP_Amenity"] = "Error"
            
        classification_results.append(result_row)

    print("\n--- Processing Complete ---")
    output_excel = f"{session_dir}/siglip_osm_classifications.xlsx"
    df_results = pd.DataFrame(classification_results)
    
    cols = ['Building_ID', 'OSM_Building', 'SigLIP_Building', 'SigLIP_B_Conf', 
            'OSM_Amenity', 'SigLIP_Amenity', 'SigLIP_A_Conf']
    df_results = df_results[cols]
    
    df_results.to_excel(output_excel, index=False)
    print(f"Results successfully saved to {output_excel}")

if __name__ == "__main__":
    main()