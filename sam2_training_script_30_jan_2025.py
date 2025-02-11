import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW


image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Training_data_for_sam2_/train_data_for_sam2_contains_only_one_copy_first_4_city/images/'
mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Training_data_for_sam2_/train_data_for_sam2_contains_only_one_copy_first_4_city/masks/'
txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Training_data_for_sam2_/train_data_for_sam2_contains_only_one_copy_first_4_city/txt_files/'

sam2_checkpoint = "sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_l.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

def load_image_mask_and_points(image_dir, mask_dir, txt_dir):
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(len(image_files))

    # Iterate over all image files
    for image_filename in image_files:
        print("length of images :", image_filename)
        
        # Construct corresponding mask and txt file paths
        mask_filename = os.path.join(mask_dir, image_filename)  # Assuming mask has the same name as the image
        # print("Mask filename:", mask_filename)
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
      
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Read image
        ann_map = cv2.imread(mask_filename)  # Read annotation (mask)
        
        # Read points from the text file
        with open(txt_filename, 'r') as f:
            points = [list(map(int, line.strip().split(','))) for line in f.readlines()]
            print("Points:", points)

        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)))

        # Yield the processed image, mask, points, and labels
        yield Img, np.array(ann_map), np.array(points), np.ones([len(points), 1])
    
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)

predictor.model.image_encoder.train(False)

predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

no_of_epochs = 100 # @param


# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2_22_jan_december_2024_8_accumulations_step"

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) 
accumulation_steps = 8  # Number of steps to accumulate gradients before updating

checkpoint_path = f"{FINE_TUNED_MODEL_NAME}_latest.torch"

# Load checkpoint if available
def load_checkpoint(model, optimizer, checkpoint_path=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        mean_iou = checkpoint.get('mean_iou', 0)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch, mean_iou = 1, 0
    return start_epoch, mean_iou

start_epoch, mean_iou = load_checkpoint(predictor.model, optimizer, checkpoint_path)

# Training loop
for epoch in range(start_epoch, no_of_epochs + 1):
    data_generator_21 = load_image_mask_and_points(image_dir, mask_dir, txt_dir)
    
    for idx, (image, mask, input_point, input_label) in enumerate(data_generator_21):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        input_label = input_label.flatten()
        
        if image is None or mask is None or np.any(input_label == 0):
            continue
        if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
            continue
        if input_point.size == 0 or input_label.size == 0:
            continue

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue
        
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )
        
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda().unsqueeze(0)
        gt_mask = torch.sigmoid(gt_mask) > 0.5
        gt_mask = gt_mask.to(torch.float32)
        
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = F.binary_cross_entropy(prd_mask, gt_mask)
        
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.09
        loss = loss / accumulation_steps
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        
        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    scheduler.step()
    
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': predictor.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mean_iou': mean_iou,
        }, f"{FINE_TUNED_MODEL_NAME}_{epoch}.torch")
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': predictor.model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'mean_iou': mean_iou,
        # }, checkpoint_path)
    
    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
    
    if epoch % 5 == 0:
        print(f"Step {epoch}: Accuracy (IoU) = {mean_iou}")
    
    print(f"Epoch {epoch} is completed!")
