import os
import json
import torch
import numpy as np
import cv2
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# 1. Custom Dataset for COCO JSON Format
# -------------------------------------------------------------------
class PanoramaInstanceDataset(Dataset):
    def __init__(self, json_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load COCO JSON
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
            
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Map image_id to its annotations for fast lookup
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Read image (convert BGR to RGB)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}. Please check your images folder.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Fetch annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            # 1. Parse Bounding Box (COCO is [x, y, w, h] -> PyTorch is [xmin, ymin, xmax, ymax])
            xmin, ymin, w, h = ann['bbox']
            xmax, ymax = xmin + w, ymin + h
            
            # Filter out degenerate boxes
            if w <= 0 or h <= 0:
                continue
                
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])
            
            # 2. Parse Segmentation Polygon to Binary Mask
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
            for poly in ann['segmentation']:
                # Reshape to (N, 2)
                poly_np = np.array(poly).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly_np], 1)
            masks.append(mask)

        # Convert to Tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Handle empty images
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, img_info['height'], img_info['width']), dtype=torch.uint8)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.empty((0,))
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd
        }

        # Convert image to FloatTensor in range [0, 1]
        img = F.to_tensor(img)

        return img, target

# -------------------------------------------------------------------
# 2. Model Initialization (Mask R-CNN)
# -------------------------------------------------------------------
def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    # Using the updated API for torchvision >= 0.13 (and 0.25+)
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    # 1. Update the Box Predictor Head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Update the Mask Predictor Head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# Collate function needed for variable number of objects per image
def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------------------------------------------------------
# 3. Training Loop
# -------------------------------------------------------------------
def main():
    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Paths
    JSON_PATH = "annotation-panorama-instance-train.json"
    IMG_DIR = "image-panorama-train" 

    # Dataset & DataLoader
    dataset = PanoramaInstanceDataset(json_file=JSON_PATH, img_dir=IMG_DIR)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=8, # Adjust based on your GPU VRAM. 2 is safe for 512x1024
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn
    )

    # Extract number of unique categories from dataset to set num_classes
    # COCO background is 0, so num_classes = max_category_id + 1
    categories = [cat['category_id'] for cat in dataset.annotations]
    num_classes = max(categories) + 1  
    print(f"Detected {num_classes} classes (including background).")

    # Initialize Model
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0001)
    
    # Learning Rate Scheduler
    num_epochs = 5
    steps_per_epoch = len(data_loader) 
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0002,                    
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs
    )
    
    scaler = torch.amp.GradScaler()

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

    # 1. Scale the loss and calculate gradients
            scaler.scale(losses).backward()
            
            # 2. Unscale gradients before clipping so the threshold makes sense
            scaler.unscale_(optimizer)
            
            # 3. GRADIENT CLIPPING: Prevent exploding gradients (NaNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            # Save the current scale before we step
            scale_before = scaler.get_scale()
            
            # 4. Step the optimizer
            scaler.step(optimizer)
            scaler.update()

            # ONLY step the scheduler if the scaler didn't skip the optimizer.step()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                lr_scheduler.step()

            train_loss += losses.item()
            
            if i % 10 == 0:
                # Extract each individual loss from the dictionary
                l_cls = loss_dict['loss_classifier'].item()
                l_box = loss_dict['loss_box_reg'].item()
                l_mask = loss_dict['loss_mask'].item()
                l_obj = loss_dict['loss_objectness'].item()
                l_rpn = loss_dict['loss_rpn_box_reg'].item()
                
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] | Batch [{i}/{len(data_loader)}] | Total Loss: {losses.item():.4f}\n"
                    f"    -> Cls: {l_cls:.4f} | Box: {l_box:.4f} | Mask: {l_mask:.4f} | Obj: {l_obj:.4f} | RPN Box: {l_rpn:.4f}"
                )

        print(f"--- Epoch [{epoch+1}/{num_epochs}] Average Loss: {train_loss/len(data_loader):.4f} ---")
    # Save the trained model
    torch.save(model.state_dict(), "panorama_maskrcnn_model.pth")
    print("Training complete! Model saved to 'panorama_maskrcnn_model.pth'.")

if __name__ == "__main__":
    main()