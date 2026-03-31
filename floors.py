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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from tkinter import filedialog
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import streetview
from datetime import datetime

# SAM 3 Imports
from sklearn.cluster import DBSCAN
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# External dependencies
from equilib import Equi2Pers
from rectification import (
    compute_edgelets, ransac_vanishing_point, reestimate_model, 
    remove_inliers, compute_homography_and_warp
)
from skimage import io

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

    # Ensure box_width is strictly positive
    box_width = max(max_col - min_col, 1)
    
    # Calculate FOVs, but clamp them to 160 degrees max! 
    # Perspective projection breaks down at >= 180 degrees
    fov_x_deg = min((box_width / W) * 360.0, 160.0)

    angle_top = (0.5 - min_row / H) * 180.0
    angle_bottom = (0.5 - max_row / H) * 180.0
    max_y_angle = max(abs(angle_top), abs(angle_bottom))
    fov_y_deg = min(max_y_angle * 2.0 * 1.1, 160.0)

    # Ensure out_width is at least 10 pixels
    out_width = max(int(box_width * 1.5), 10) 
    
    tan_y = np.tan(np.radians(max(fov_y_deg, 1) / 2.0))
    tan_x = np.tan(np.radians(max(fov_x_deg, 1) / 2.0))
    
    # Ensure out_height is at least 10 pixels and mathematically positive
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

def orthorectify_facade(image_path):
    image = io.imread(image_path)
    edgelets1 = compute_edgelets(image)
    vp1 = ransac_vanishing_point(edgelets1, num_ransac_iter=2000, threshold_inlier=5)
    edgelets2 = remove_inliers(vp1, edgelets1, 10)
    vp2 = ransac_vanishing_point(edgelets2, num_ransac_iter=2000, threshold_inlier=5)
    warped_img = compute_homography_and_warp(image, vp1, vp2)
    return warped_img

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.5])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.5])
    if len(mask.shape) == 3:
        mask = mask
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def count_window_rows(window_masks, image_height):
    if len(window_masks) == 0:
        return 0, []
    y_centers = []
    heights = []
    
    for mask in window_masks:
        mask_2d = mask if len(mask.shape) == 3 else mask
        coords = np.where(mask_2d)
        if len(coords[-2]) == 0:
            continue
        y_coords = coords[-2]
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        heights.append(max_y - min_y)
        y_centers.append((min_y + max_y) / 2.0)

    if not y_centers:
        return 0, []

    median_height = np.median(heights)
    y_centers_2d = np.array(y_centers).reshape(-1, 1)
    eps_threshold = max(1.0, median_height)
    clustering = DBSCAN(eps=eps_threshold, min_samples=1).fit(y_centers_2d)

    row_centers = []
    unique_labels = set(clustering.labels_)
    for cluster_id in unique_labels:
        if cluster_id != -1: 
            cluster_points = y_centers_2d[clustering.labels_ == cluster_id]
            row_centers.append(float(np.mean(cluster_points)))

    row_centers.sort()
    return len(row_centers), row_centers

def detect_floors(image_path, processor, save_plot_path=None):
    image = Image.open(image_path).convert("RGB")
    image_height, image_width = image.height, image.width
    inference_state = processor.set_image(image)
    
    if save_plot_path:
        plt.figure(figsize=(12, 10))
        plt.imshow(image)
    
    negative_prompts = ["car window", "car", "vase", "tree", "bush", "sidewalk", "person", "animal", "signboard", "streetlight", "traffic light", "furniture"]
    global_exclusion_mask = np.zeros((image_height, image_width), dtype=bool)
    for neg_prompt in negative_prompts:
        neg_masks = processor.set_text_prompt(state=inference_state, prompt=neg_prompt).get('masks', [])
        if torch.is_tensor(neg_masks): neg_masks = neg_masks.cpu().numpy()
        for m in neg_masks:
            m_2d = m if len(m.shape) == 3 else m
            global_exclusion_mask = np.logical_or(global_exclusion_mask, m_2d)

    prompts = ["building window", "building door"]
    all_window_masks = [] 
    for prompt in prompts:
        results = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = results.get('masks', [])
        if torch.is_tensor(masks): masks = masks.cpu().numpy()
        
        for i in range(len(masks)):
            mask = masks[i]
            mask_2d = mask if len(mask.shape) == 3 else mask
            overlap_ratio = np.sum(np.logical_and(mask_2d, global_exclusion_mask)) / np.sum(mask_2d)
            if overlap_ratio <= 0.05:
                all_window_masks.append(mask)
                if save_plot_path: show_mask(mask, plt.gca(), random_color=True)

    row_count, row_y_coords = count_window_rows(all_window_masks, image_height)
    
    if save_plot_path:
        for y_coord in row_y_coords:
            plt.axhline(y=y_coord, color='red', linestyle='--', linewidth=2, alpha=0.8)
        plt.axis('off')
        plt.title(f"Floors Detected: {row_count}")
        plt.tight_layout()
        plt.savefig(save_plot_path)
        plt.close()

    return row_count, row_y_coords


def main():
    GOOGLE_MAPS_API_KEY = input("Enter your Google Maps API Key: ").strip()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # --- NEW: Create Session Directory ---
    session_dir = datetime.now().strftime("session_%Y%m%d_%H%M%S") + "_floors_output"
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
    print("Loading Mask R-CNN model...")
    mask_model = get_model_instance_segmentation(num_classes=2)
    mask_model.load_state_dict(torch.load("weights/panorama_maskrcnn_model.pth", map_location=device))
    mask_model.to(device)
    mask_model.eval()

    print("Loading SAM 3 model...")
    sam_checkpoint = "weights/sam3.pt" 
    sam_model = build_sam3_image_model(checkpoint_path=sam_checkpoint)
    sam_model.to(device)
    sam_processor = Sam3Processor(sam_model, confidence_threshold=0.4)

    # 3. Authenticate Google API
    headers = {'Content-Type': 'application/json'}
    data = """{"mapType": "streetview","language": "en-US", "region": "US"}"""
    response = requests.post(f"https://tile.googleapis.com/v1/createSession?key={GOOGLE_MAPS_API_KEY}", headers=headers, data=data)
    SESSION_TOKEN = response.json()['session']
    
    floor_results = []
    
    # 4. Process Each Building
    for idx, row in buildings.iterrows():
        osm_id = str(row['osm_id'])
        bldg_lon, bldg_lat = row.geometry.centroid.x, row.geometry.centroid.y
        print(f"\n--- Processing Shapefile Building {osm_id} ---")
        
        final_floor_count = "Failed/No Pano" 
        pano_id = "UNKNOWN"

        panos = streetview.panoids(lat=bldg_lat, lon=bldg_lon)
        if not panos: 
            floor_results.append({"Building_ID": f"{osm_id}_{pano_id}", "Floor_Count": final_floor_count})
            continue
        
        try:
            panos_df = pd.DataFrame(panos)
            pano_id = panos_df.sort_values(by='year', ascending=False).iloc[0]["panoid"]
        except KeyError:
            pano_id = panos[0]["panoid"]
        
        meta = get_pano_metadata(pano_id, GOOGLE_MAPS_API_KEY)
        if not meta: 
            floor_results.append({"Building_ID": f"{osm_id}_{pano_id}", "Floor_Count": final_floor_count})
            continue
            
        bearing = get_bearing(meta['lat'], meta['lng'], bldg_lat, bldg_lon)
        pano_heading = meta['heading'] 

        # --- EXACT DOWNLOADING AND STITCHING LOGIC (UPDATED WITH SESSION_DIR) ---
        print(f"  Downloading tiles for pano: {pano_id}...")
        
        g1 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/0/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g2 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/1/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g3 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/2/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g4 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/3/0?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g5 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/0/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g6 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/1/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g7 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/2/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")
        g8 = requests.get(f"https://tile.googleapis.com/v1/streetview/tiles/2/3/1?session={SESSION_TOKEN}&key={GOOGLE_MAPS_API_KEY}&panoId={pano_id}")

        with open(f"{session_dir}/{pano_id}_0.jpg", 'wb') as f: f.write(g1.content)
        with open(f"{session_dir}/{pano_id}_1.jpg", 'wb') as f: f.write(g2.content)
        with open(f"{session_dir}/{pano_id}_2.jpg", 'wb') as f: f.write(g3.content)
        with open(f"{session_dir}/{pano_id}_3.jpg", 'wb') as f: f.write(g4.content)
        with open(f"{session_dir}/{pano_id}_4.jpg", 'wb') as f: f.write(g5.content)
        with open(f"{session_dir}/{pano_id}_5.jpg", 'wb') as f: f.write(g6.content)
        with open(f"{session_dir}/{pano_id}_6.jpg", 'wb') as f: f.write(g7.content)
        with open(f"{session_dir}/{pano_id}_7.jpg", 'wb') as f: f.write(g8.content)

        images_r1 = [Image.open(x) for x in [f"{session_dir}/{pano_id}_0.jpg", f"{session_dir}/{pano_id}_1.jpg", f"{session_dir}/{pano_id}_2.jpg", f"{session_dir}/{pano_id}_3.jpg"]]
        images_r2 = [Image.open(x) for x in [f"{session_dir}/{pano_id}_4.jpg", f"{session_dir}/{pano_id}_5.jpg", f"{session_dir}/{pano_id}_6.jpg", f"{session_dir}/{pano_id}_7.jpg"]]
        
        widths, heights = zip(*(i.size for i in images_r1))
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im_r1 = Image.new('RGB', (total_width, max_height))
        new_im_r2 = Image.new('RGB', (total_width, max_height))
        new_im = Image.new('RGB', (total_width, 2*max_height))

        x_offset1 = 0
        x_offset2 = 0
        y_offset = 0

        for im in images_r1:
            new_im_r1.paste(im, (x_offset1,0))
            x_offset1 += im.size[0]

        for im in images_r2:
            new_im_r2.paste(im, (x_offset2,0))
            x_offset2 += im.size[0]

        ims = [new_im_r1, new_im_r2]

        for img in ims:
            new_im.paste(img, (0, y_offset))
            y_offset += max_height

        stitched_path = f"{session_dir}/{pano_id}.jpg"
        new_im.save(stitched_path)
        
        for i in range(8):
            os.remove(f"{session_dir}/{pano_id}_{i}.jpg")
        # -------------------------------------------------------------
        
        orig_img_cv = cv2.imread(stitched_path)
        if orig_img_cv is None: 
            floor_results.append({"Building_ID": f"{osm_id}_{pano_id}", "Floor_Count": final_floor_count})
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

        final_floor_count = "AI Segment Failed"

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
                floor_results.append({
                    "Building_ID": f"{osm_id}_{pano_id}", 
                    "Floor_Count": "Extraction Error"
                })
                continue # Skip the rest of the loop for this building

            temp_facade_name = f"{session_dir}/temp_facade_{pano_id}.jpg"
            cv2.imwrite(temp_facade_name, cv2.cvtColor(facade_img, cv2.COLOR_RGB2BGR))
            
            MAX_RETRIES = 3
            success = False
            
            for attempt in range(MAX_RETRIES):
                print(f"  -> Orthorectification Attempt {attempt + 1}/{MAX_RETRIES}...")
                try:
                    rectified_img = orthorectify_facade(temp_facade_name)
                    temp_rectified_name = f"{session_dir}/temp_rectified_{pano_id}.jpg"
                    plt.imsave(temp_rectified_name, rectified_img)
                    
                    plot_output = f"{session_dir}/final_annotated_{osm_id}_{pano_id}.jpg"
                    floor_count, _ = detect_floors(temp_rectified_name, sam_processor, save_plot_path=plot_output)
                    
                    if floor_count > 0:
                        print(f"  -> Success! Validated {floor_count} floors. Saved to {plot_output}")
                        final_floor_count = floor_count 
                        success = True
                        if os.path.exists(temp_rectified_name): os.remove(temp_rectified_name)
                        break 
                    else:
                        print("  -> Validation failed (0 windows detected). RANSAC likely generated a bad warp.")
                        
                except Exception as e:
                    print(f"  -> Rectification crashed on attempt {attempt + 1}: {e}")
            
            if not success:
                print(f"  -> Giving up on building {osm_id} after {MAX_RETRIES} bad rectifications.")
                final_floor_count = "0 (Warp Failed)"
                
            if os.path.exists(temp_facade_name): os.remove(temp_facade_name)
            
        floor_results.append({
            "Building_ID": f"{osm_id}_{pano_id}", 
            "Floor_Count": final_floor_count
        })

    # 5. Export Results (Saved directly to the session folder)
    print("\n--- Processing Complete ---")
    output_excel = f"{session_dir}/building_floor_counts.xlsx"
    df_results = pd.DataFrame(floor_results)
    df_results.to_excel(output_excel, index=False)
    print(f"Results successfully saved to {output_excel}")

if __name__ == "__main__":
    main()