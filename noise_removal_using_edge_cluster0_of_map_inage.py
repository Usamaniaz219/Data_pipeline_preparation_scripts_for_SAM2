import os
import cv2
import numpy as np



#####################################################################################################################
# Step1 : Canny Edge Detection for the Edge detection

image_path = "/home/usama/test_data_for_sam2_26_feb_2025/input_images/WE3474.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

edge =  cv2.adaptiveThreshold(image, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
# t_lower = 50  # Lower Threshold 
# t_upper = 150  # Upper threshold 
  
# # Applying the Canny Edge filter 
# edge = cv2.Canny(image, t_lower, t_upper) 
cv2.imwrite("we3474_edges.jpg",edge)
  


###############################################################################################################################



##############################################################################################################################
# Step2 : Draw the edges detected on the cluster1

import cv2

# Load and preprocess edge image
edge_path = "/home/usama/test_images/we3474_edges.jpg"
edge_image = cv2.imread(edge_path)
edge_image_gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
_, edge_image_gray = cv2.threshold(edge_image_gray, 50, 255, cv2.THRESH_BINARY)

# Load and preprocess mask image
mask_path = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/WE3474-405b Zoning Map-page-001/WE3474-405b Zoning Map-page-001_0.jpg"
mask_image = cv2.imread(mask_path)
mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
_, mask_image_gray = cv2.threshold(mask_image_gray, 50, 255, cv2.THRESH_BINARY)

# Combine both masks
resultant_mask_image = cv2.bitwise_or(mask_image_gray, edge_image_gray)

kernel = np.ones((3, 3), np.uint8) 
# resultant_mask_image = cv2.medianBlur(resultant_mask_image,5)
resultant_mask_image = cv2.morphologyEx(resultant_mask_image, cv2.MORPH_CLOSE, kernel,iterations=1)

cv2.imwrite("resultant_mask_image.jpg", resultant_mask_image)

#########################################################################################################################################





##########################################################################################################################################
# Step3: Subtract the image generated in step2 from the target cluster in order to denoise it


import cv2
import numpy as np

# # Example binary mask (foreground = 1, background = 0)
mask = cv2.imread('/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/WE3474-405b Zoning Map-page-001/WE3474-405b Zoning Map-page-001_1.jpg', cv2.IMREAD_GRAYSCALE)

# # Example edge mask (edges = 255, non-edges = 0)
edges = cv2.imread('/home/usama/test_images/resultant_mask_image.jpg', cv2.IMREAD_GRAYSCALE)


# # Ensure the masks are binary (0 or 255) if needed
_, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

_, edges = cv2.threshold(edges,128, 255, cv2.THRESH_BINARY)

mask_to_be_updated = mask.copy()
# cv2.imshow('Original Mask', mask)
cv2.imshow('Edges', edges)
# cv2.imshow('Intersection', intersection)
cv2.imshow('Updated Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_updated = cv2.subtract(mask_to_be_updated,edges)

mask_updated = cv2.medianBlur(mask_updated,5)

# # Save or display the result
cv2.imwrite('updated_mask.png', mask_updated)


################################################################################################################################################