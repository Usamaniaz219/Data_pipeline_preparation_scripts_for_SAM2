

import os
import cv2
import numpy as np

# Path to the directory containing the mask tiles
mask_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Data_pipeline_for_sam2_31_oct_2024/ca_colma_sam2_meanshift_overlay_res/ori_masks_polygons_merged/"

# List all image files in the directory
image_files = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

for image_file in image_files:
    # Full path to the image
    image_path = os.path.join(mask_dir, image_file)
    
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is not None:
        # Apply dilation followed by erosion
        # dilated = cv2.dilate(image, kernel, iterations=3)
        # processed = cv2.erode(dilated, kernel, iterations=1)

        dilation= cv2.dilate(image, np.ones((5, 5), np.uint8), iterations=2) 
        # show_image('dilate', dilation)

        # erosion= cv2.erode(dilation, np.ones((5, 5), np.uint8), iterations=2) 
        # # show_image('erosion', erosion)

        # processed= cv2.erode(erosion, np.ones((5, 5), np.uint8), iterations=2) 

        erosion= cv2.erode(dilation, np.ones((5, 5), np.uint8), iterations=5) 
    # show_image('erosion', erosion)

        dilation2= cv2.dilate(erosion, np.ones((5, 5), np.uint8), iterations=3) 



        
        # Save the processed image back to the same file (overwrite)
        cv2.imwrite(image_path, dilation2)
    else:
        print(f"Failed to read the image: {image_file}")
