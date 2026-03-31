import os
# FIX FOR OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- 1. Model Initialization ---
def get_model_instance_segmentation(num_classes):
    """Loads the Mask R-CNN model architecture matching your saved weights."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

# --- 2. Metrics Calculation ---
def calculate_metrics(pred_mask, gt_mask):
    """Calculates Pixel Accuracy, IoU, and Dice Coefficient for binary masks."""
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    pixel_acc = (pred_mask == gt_mask).mean()
    
    if union > 0:
        iou = intersection / union
    else:
        iou = 1.0 if gt_mask.sum() == 0 and pred_mask.sum() == 0 else 0.0
        
    if (pred_mask.sum() + gt_mask.sum()) > 0:
        dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum())
    else:
        dice = 1.0 if gt_mask.sum() == 0 and pred_mask.sum() == 0 else 0.0

    return pixel_acc, iou, dice

# --- 3. Ground Truth Mask Generation ---
def generate_gt_mask(annotations, image_id, height, width):
    """Generates a combined binary semantic mask from COCO instance polygons."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if ann.get('image_id') == image_id and 'segmentation' in ann:
            for seg in ann['segmentation']:
                pts = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.fillPoly(mask, [pts], 1)
    return mask

# --- 4. Main Script ---
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running evaluation on: {device}")

    # Paths (Update these if necessary)
    JSON_PATH = "annotation-panorama-instance-test.json"
    IMG_DIR = "image-panorama-test"       
    OUTPUT_DIR = "output_semantic_metrics"   
    MODEL_PATH = "weights/panorama_maskrcnn_model.pth" # Ensure this points to your Mask R-CNN weights
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Annotations
    print("Loading JSON annotations...")
    with open(JSON_PATH, 'r') as f:
        coco_data = json.load(f)
    
    images_info = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])

    # Load Model (Mask R-CNN)
    num_classes = 2 # 0: Background, 1: Foreground
    model = get_model_instance_segmentation(num_classes)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Weights file not found at {MODEL_PATH}")
        
    model.to(device)
    model.eval() # Set to evaluation mode

    # Loss Function for Semantic Validation
    # Using BCE since we are comparing flat probability maps to binary ground truth
    criterion = nn.BCELoss()
    transform = transforms.ToTensor()

    # Store Metrics
    all_ious, all_dices, all_accs, all_losses = [], [], [], []

    print("-" * 50)
    print("Starting Inference and Visualization...")
    
    for i, img_info in enumerate(images_info):
        img_name = img_info['file_name']
        img_id = img_info['id']
        height = img_info['height']
        width = img_info['width']

        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Ground Truth Mask
        gt_mask = generate_gt_mask(annotations, img_id, height, width)

        input_tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)[0]
            
            masks = prediction['masks'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # 1. Create a continuous probability map for Loss calculation
            if len(masks) > 0:
                # Merge all predicted masks by taking the maximum probability at each pixel
                prob_map = np.max(masks[:, 0, :, :], axis=0)
            else:
                prob_map = np.zeros((height, width))

            # Calculate Semantic Test Loss
            prob_tensor = torch.tensor(prob_map, dtype=torch.float32).to(device)
            gt_float_tensor = torch.tensor(gt_mask, dtype=torch.float32).to(device)
            loss = criterion(prob_tensor, gt_float_tensor)
            all_losses.append(loss.item())

            # 2. Flatten Mask R-CNN instances into a single binary Semantic Mask
            pred_mask = np.zeros((height, width), dtype=np.uint8)
            for j, score in enumerate(scores):
                if score > 0.5: # Confidence threshold
                    m = masks[j, 0] > 0.5 # Binarize the instance mask
                    pred_mask[m] = 1

        # Calculate localized metrics
        acc, iou, dice = calculate_metrics(pred_mask, gt_mask)
        all_accs.append(acc)
        all_ious.append(iou)
        all_dices.append(dice)

        # --- Generate Side-by-Side Visualizations ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title(f"Predicted Mask (Loss: {loss.item():.4f})")
        axes[2].axis('off')

        overlay = img_rgb.copy()
        overlay[pred_mask == 1] = overlay[pred_mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
        axes[3].imshow(overlay.astype(np.uint8))
        axes[3].set_title(f"Overlay (IoU: {iou:.2f})")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{img_name}"))
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images_info)} images...")

    # --- 5. Plotting Global Metric Distributions ---
    print("-" * 50)
    print("Inference complete. Generating dataset-wide metrics charts...")

    if all_ious:
        print(f"Mean Pixel Accuracy: {np.mean(all_accs):.4f}")
        print(f"Mean IoU (mIoU): {np.mean(all_ious):.4f}")
        print(f"Mean Dice Score: {np.mean(all_dices):.4f}")
        print(f"Average Test Loss (BCE): {np.mean(all_losses):.4f}")

        plt.style.use('ggplot')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].hist(all_ious, bins=20, color='royalblue', edgecolor='black')
        axes[0, 0].set_title('IoU Score Distribution')
        axes[0, 0].set_xlabel('Intersection over Union (IoU)')
        axes[0, 0].set_ylabel('Image Count')

        axes[1, 0].hist(all_dices, bins=20, color='mediumseagreen', edgecolor='black')
        axes[1, 0].set_title('Dice Coefficient Distribution')
        axes[1, 0].set_xlabel('Dice Coefficient')
        axes[1, 0].set_ylabel('Image Count')

        axes[0, 1].boxplot([all_ious, all_dices, all_accs], labels=['IoU', 'Dice', 'Pixel Accuracy'])
        axes[0, 1].set_title('Model Performance Metrics')
        axes[0, 1].set_ylabel('Score')

        axes[1, 1].plot(range(1, len(all_losses) + 1), all_losses, marker='o', color='crimson', linestyle='-')
        axes[1, 1].set_title('Semantic Test Loss (BCE) per Image')
        axes[1, 1].set_xlabel('Image Index')
        axes[1, 1].set_ylabel('Binary Cross-Entropy Loss')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "overall_semantic_metrics.png"))
        plt.close(fig)
        
        print(f"Saved dataset metrics to: {OUTPUT_DIR}/overall_semantic_metrics.png")
    else:
        print("No valid images processed to calculate overall metrics.")

if __name__ == "__main__":
    main()