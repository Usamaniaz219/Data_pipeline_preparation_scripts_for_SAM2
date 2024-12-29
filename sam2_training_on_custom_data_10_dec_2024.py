
import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import torch.nn.functional as F


image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_for_sam2_contains_only_one_copy/images/'
mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_for_sam2_contains_only_one_copy/masks/'
txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_for_sam2_contains_only_one_copy/txt_files/'


sam2_checkpoint = "sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
model_cfg = "sam2_hiera_l.yaml" # @param ["sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"]

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
        print("Text filename:", txt_filename)
        # Read image and mask
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Read image
        ann_map = cv2.imread(mask_filename)  # Read annotation (mask)
        # ann_map = cv2.cvtColor(ann_map, cv2. COLOR_BGR2GRAY)
        # _,ann_map = cv2.threshold(ann_map,0,255,cv2.THRESH_BINARY)
        
        # Read points from the text file
        with open(txt_filename, 'r') as f:
            points = [list(map(int, line.strip().split(','))) for line in f.readlines()]
            # print("Points:", points)

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

# Mix precision.
# scaler = torch.cuda.amp.GradScaler()

# No. of steps to train the model.
no_of_epochs = 100 # @param


# Fine-tuned model name.
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2_9_december_2024"



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # 500 , 250, gamma = 0.1
accumulation_steps = 4  # Number of steps to accumulate gradients before updating


for epoch in range(1, no_of_epochs + 1):
    data_generator_21 = load_image_mask_and_points(image_dir, mask_dir, txt_dir)
    for idx , (image, mask, input_point, input_label) in enumerate(data_generator_21):
        mask = cv2.cvtColor(mask, cv2. COLOR_BGR2GRAY)
        _,mask = cv2.threshold(mask,128,255,cv2.THRESH_BINARY)

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

        # batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
        ]
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

        # gt_mask = torch.tensor(mask).cuda()
        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        gt_mask_copy = gt_mask.clone()
        gt_mask = gt_mask.unsqueeze(0)
        
        gt_mask_ev = gt_mask_copy.cpu()
        gt_mask_ev = gt_mask_ev.numpy()
        cv2.imwrite("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/res/gt_mask_ev.jpg",gt_mask_ev)
        # print("ground truth mask",gt_mask)
        # gt_mask = gt_mask.permute(2, 0, 1)

        gt_mask = torch.sigmoid(gt_mask)
        gt_mask = gt_mask>0.5
        gt_mask = gt_mask.to(torch.float32)
        print("type of gt mask",type(gt_mask))


        
        
        
        # def binarize_tensor(tensor, threshold=0):
        #     return (tensor > threshold).int()
        # binarize_tensor(gt_mask, threshold=0)

        def is_binary_tensor(tensor):
            return torch.all((tensor == 0) | (tensor == 1))
        print("Is binary tensor",is_binary_tensor(gt_mask)) 
        
        print("predicted",prd_masks[:,0].shape)
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        prd_mask_p = prd_mask.clone()
        # prd_mask_p = ((prd_mask_p > 0.5)==1)
        prd_mask_p = (prd_mask_p > 0.5).cpu().numpy()
        prd_mask_p = prd_mask_p.astype(np.uint8)  # Convert to uint8
        prd_mask_p *= 255  # Scale values to [0, 255]

        # prd_mask_p = (prd_mask_p * 255).astype(np.uint8)  # Scale if necessary
        print(prd_mask_p.shape, prd_mask_p.dtype)
        prd_mask_p = np.squeeze(prd_mask_p)

        # prd_mask_p = prd_mask_p.cpu()
        cv2.imwrite("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/res/prd_mask_ev.jpg",prd_mask_p)
        
        # seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.0001)).mean()
        # seg_loss = F.binary_cross_entropy(prd_mask, gt_mask)
        seg_loss = F.binary_cross_entropy(prd_mask, gt_mask)
        print("segmentation loss",seg_loss)
        
        
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        prd_mask_ = prd_mask>0.5
        # inter_1 = torch.logical_and(gt_mask,prd_mask_).sum(1).sum(1)
        # # print("intersection value",inter)
        # union = (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        # # print("union",union)
        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
        print("iou value",iou)



        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        # loss = seg_loss + score_loss * 0.05
        loss = seg_loss + score_loss * 0.09
        print("loss value",loss)

        # Apply gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
        idx_1 = idx+1
        if idx_1 % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Update scheduler
    scheduler.step()

    if epoch % 10 == 0:
        FINE_TUNED_MODEL = FINE_TUNED_MODEL_NAME + "_" + str(epoch) + ".torch"
        torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

    if epoch == 1:
        mean_iou = 0

    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

    if epoch % 5 == 0:
        print("Step " + str(epoch) + ":\t", "Accuracy (IoU) = ", mean_iou)
    print(f"Epoch{epoch} is completed!")









































