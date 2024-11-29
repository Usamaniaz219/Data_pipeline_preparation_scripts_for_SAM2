import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


image_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/data_test/images/"
txt_files_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/data_test/txt_files/"

def load_image_and_points(image_dir, txt_dir):
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
        # print("Processing image:", image_filename)
        
        # Construct corresponding mask and txt file paths
    
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
        
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
            print(points)
        

        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
    
        # Yield the processed image, mask, points, and labels
        yield Img, np.array(points)


data_generator_41  = load_image_and_points(image_path,txt_files_path)

for img1,input_points_21 in data_generator_41:
    # Load the fine-tuned model
    sam2_checkpoint = "segment-anything-2/sam2_hiera_small.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
    model_cfg = "sam2_hiera_s.yaml" 

    FINE_TUNED_MODEL_WEIGHTS = "segment-anything-2/fine_tuned_sam2_100.torch"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    print(sam2_model)
    


    # Build net and load weights
    predictor = SAM2ImagePredictor(sam2_model)
    print("predictor",predictor)
    print(dir(predictor))
    predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    # Perform inference and predict masks
    with torch.no_grad():
        predictor.set_image(img1)
        masks, scores, logits = predictor.predict(
            point_coords=input_points_21,
            point_labels=np.ones([input_points_21.shape[0], 1])
        )
    
    # np_masks = np.array(masks[:, 0])

    # print("np_masks",np_masks.shape)
    if scores.ndim == 1:
        np_masks = np.array(masks)
        np_scores = scores  # If it's 1D, use it directly
        print("np scores",np_scores)
    else:
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0] 
        print("np scores",np_scores)
# print("np scores",np_scores)
    sorted_masks = np_masks[np.argsort(np_scores)][::-1]
    print("sorted masks",len(sorted_masks))
    # Initialize segmentation map and occupancy mask
    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)
    # occupancy_mask = np.zeros_like(sorted_masks[0], dtype=bool)
    # for i in range(sorted_masks.shape[0]):
    mask_11 = sorted_masks[0]
    mask_bool = mask_11.astype(bool)
    # mask_bool[occupancy_mask] = False  # Set overlapping areas to False in the mask
    seg_map[mask_bool] = 1  # Use boolean mask to index seg_map
    # occupancy_mask[mask_bool] = True  # Update occupancy_mask
    print("seg shape",seg_map.shape)
    seg_map = seg_map.astype(np.uint8)
    _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)
    cv2.imwrite("seg_mask_1.jpg",seg_map)
  

