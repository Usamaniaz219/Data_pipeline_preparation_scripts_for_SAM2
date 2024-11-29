
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output_1/images/'
mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output_1/masks/'
txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output_1/txt_files/'


sam2_checkpoint = "segment-anything-2/sam2_hiera_small.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_s.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

def load_image_mask_and_points(image_dir, mask_dir, txt_dir):
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
        print("Processing image:", image_filename)
        
        # Construct corresponding mask and txt file paths
        mask_filename = os.path.join(mask_dir, image_filename)  # Assuming mask has the same name as the image
        print("Mask filename:", mask_filename)
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
        
        # Read image and mask
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
# Train mask decoder.
predictor.model.sam_mask_decoder.train(True)

# Train prompt encoder.
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer.
optimizer=torch.optim.AdamW(params=predictor.model.parameters(),lr=0.0001,weight_decay=1e-4) #1e-5, weight_decay = 4e-5

# Mix precision.
scaler = torch.cuda.amp.GradScaler()

# No. of steps to train the model.
no_of_epochs = 100 # @param


# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2"

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2) # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating

for epoch in range(1, no_of_epochs + 1):
   with torch.cuda.amp.autocast():
      #  image, mask, input_point, input_label = load_image_mask_and_points(image_dir, mask_dir, txt_dir)
       data_generator_21 = load_image_mask_and_points(image_dir, mask_dir, txt_dir)
       for image, mask, input_point, input_label in data_generator_21:

          input_label = input_label.flatten()

        #   if image is None or mask is None or input_label == 0:
        #       continue
          if image is None or mask is None or np.any(input_label == 0):
              continue

          #  input_label = np.array(num_masks)
          if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
              continue

          if input_point.size == 0 or input_label.size == 0:
              continue

          predictor.set_image(image)
          mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
          if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
              continue

          sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
              points=(unnorm_coords, labels), boxes=None, masks=None,
          )
          print("saprse embedding",sparse_embeddings.shape)
          print("dense embedding shape",dense_embeddings.shape)

          batched_mode = unnorm_coords.shape[0] > 1
          high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
          low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
              image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
              image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=True,
              repeat_image=batched_mode,
              high_res_features=high_res_features,
          )
          prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

          gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
          gt_mask =gt_mask.permute(2, 0, 1)
        #   print("gt mask shape",gt_mask.shape)
          prd_mask = torch.sigmoid(prd_masks[:, 0])
        #   print("prd mask shape",prd_mask.shape)
          seg_loss = (-gt_mask * torch.log(prd_mask + 0.000001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean()

          inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
          iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
          score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
          loss = seg_loss + score_loss * 0.05

          # Apply gradient accumulation
          loss = loss / accumulation_steps
          scaler.scale(loss).backward()

          # Clip gradients
          torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

          if epoch % accumulation_steps == 0:
              scaler.step(optimizer)
              scaler.update()
              predictor.model.zero_grad()

          # Update scheduler
          scheduler.step()

          if epoch % 10 == 0:
              FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(epoch) + ".torch"
              torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

          if epoch == 1:
              mean_iou = 0

          mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

          if epoch % 100 == 0:
              print("Step " + str(epoch) + ":\t", "Accuracy (IoU) = ", mean_iou)