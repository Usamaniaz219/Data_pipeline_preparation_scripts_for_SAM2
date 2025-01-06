# Inference Code #
#######################

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


image_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/evaluations_samples_data_for_sam2_2_jan_2025/30_samples_automated_multiple_points_data_2_jan_2025_11_copy/images/"
txt_files_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/evaluations_samples_data_for_sam2_2_jan_2025/30_samples_automated_multiple_points_data_2_jan_2025_11_copy/txt_files/"
background_txt_files = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/evaluations_samples_data_for_sam2_2_jan_2025/30_samples_automated_multiple_points_data_2_jan_2025_11_copy/background_txt_files/"


def load_image_and_both_fore_background_points(image_dir, txt_dir,background_txt_dir):
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
        print("Processing image:", image_filename)
        
        # Construct corresponding mask and txt file paths
    
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt

        background_txt_filename = os.path.join(background_txt_dir,os.path.splitext(image_filename)[0]+'.txt')

        
        # Read image and mask
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Read image
        
        
        # Read points from the text file
        with open(txt_filename, 'r') as f:
            # points = [[list(map(int, line.strip().split(',')))] for line in f.readlines()]
            
            points = []
            for line in f:
                line = line.strip()
                if line:  # Only process non-empty lines
                    points.append([list(map(int, line.split(',')))])
            print("points",points)
        with open(background_txt_filename, 'r') as f:
             background_points = []
             for line in f:
                 line = line.strip()
                 if line:  # Only process non-empty lines
                     background_points.append([list(map(int, line.split(',')))])
             print("background_points",background_points)
            
        

        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1]*r), int(Img.shape[0]*r)))

        all_points = np.array(points + background_points)
        
        all_labels = np.concatenate([np.ones((len(points), 1)), np.zeros((len(background_points), 1))])

    
        # Yield the processed image, mask, points, and labels
        yield Img, all_points,all_labels, image_filename


# data_generator_41  = load_image_and_points(image_path,txt_files_path)

data_generator_41  = load_image_and_both_fore_background_points(image_path, txt_files_path,background_txt_files)

for idx,(img1,input_points_21,all_labels, img_name) in enumerate(data_generator_41):
    all_labels_flattened = all_labels.flatten()
    print("all_labels flattened",all_labels_flattened)
    # Load the fine-tuned model
    sam2_checkpoint = "sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
    model_cfg = "sam2_hiera_l.yaml" 

    FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_1_jan_2025_with_8_accumulation_steps_200.torch"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

    predictor = SAM2ImagePredictor(sam2_model)
  
    predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    # Perform inference and predict masks
    with torch.no_grad():
        predictor.set_image(img1)
        masks, scores, logits = predictor.predict(
            point_coords=input_points_21,
            point_labels=all_labels)
    
    
    # np_masks = np.array(masks[:, 0])

    # print("np_masks",np_masks.shape)
    if scores.ndim == 1:
        np_masks = np.array(masks)
        np_scores = scores  # If it's 1D, use it directly
        # print("np scores",np_scores)
    else:
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0] 


    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)

    mask_11 = sorted_masks
    mask_bool = mask_11.astype(bool)
    # seg_map[mask_bool] = 1  
    
    
    for i in range(mask_bool.shape[0]):
        mask = mask_bool[i]
        seg_map[mask]=1

    seg_map = seg_map.astype(np.uint8)
    _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)

    cv2.imwrite(f"/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/1_jan_2025_Sam2_checkpoints_res/1_jan_checkpoint_results_200_epochs_6_jan_2025/Results_with_4_foreground_and_background_points_automatic/{img_name}",seg_map)
    print(idx)