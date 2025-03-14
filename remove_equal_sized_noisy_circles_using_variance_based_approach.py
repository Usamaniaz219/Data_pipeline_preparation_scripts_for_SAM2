

import cv2
import numpy as np
import statistics
import os

def denoise_mask(mask):
    """
    Removes components in the mask where the variance of pixel coordinates exceeds a threshold.
    
    :param mask: Binary mask (numpy array of shape HxW, dtype=uint8, values 0 or 255)
    :param variance_threshold: Variance threshold for removing noisy components.
    :return: Denoised binary mask.
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # print("Number of labels:", num_labels)

    var_list = []
    denoised_mask = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):  # Skip background label (0)
        points = np.column_stack(np.where(labels == label))
        
        # Compute variance of the points
        var_x = np.var(points[:, 0])  
        print(f"varience of x {var_x}")
        var_y = np.var(points[:, 1])  
        print(f"varience of y {var_y}")
        total_variance = var_x + var_y
        # print(f"Label {label} - Variance: {total_variance}")

        # Keep track of valid variances
        # if total_variance < variance_threshold:
        var_list.append(total_variance)

    # Compute median variance (ensure list is not empty)
    if var_list:
        median_variance = statistics.median(var_list)
        print("Median Variance:", median_variance)
    else:
        median_variance = 0  # Fallback

    # Apply filtering based on median variance
    for label_1 in range(1, num_labels): 
        # print("label 1 is ",label_1)

        
        # if label_1==6:
        points_1 = np.column_stack(np.where(labels == label_1))
        var_x_1 = np.var(points_1[:, 0])  
        var_y_1 = np.var(points_1[:, 1])  
        total_variance_1 = var_x_1 + var_y_1
        print("total variance",total_variance_1)

        if total_variance_1 > median_variance+100:  
            denoised_mask[labels == label_1] = 255  # Retain valid components

    return denoised_mask


mask_input_directory = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo115/denoised_masks_polygons_merged/demo115/masks/demo115/"
denoise_mask_output_directory = "/home/usama/demo161_test_code/varience_and_size_based_noise_removal/demo115_merged_outputs/"

for filename in os.listdir(mask_input_directory):
    if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):
        print(f"filename path {filename}")
        target_mask_path = os.path.join(mask_input_directory, filename)
        mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
        mask_thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
        noise_mask_to_be_removed= denoise_mask(mask_thresh)
        # denoised_mask = cv2.subtract(mask_thresh,noise_mask_to_be_removed)



        # Save denoised mask with the same name in "denoised_masks" folder
        denoised_output_path = os.path.join(denoise_mask_output_directory, filename)
        # denoised_mask = cv2.erode(denoised_mask,kernel,iterations=1)
        cv2.imwrite(denoised_output_path, noise_mask_to_be_removed)
        # print(f"Denoised mask saved: {denoised_output_path}")


# mask = cv2.imread("rough_outputs/denoised_mask_6.jpg", cv2.IMREAD_GRAYSCALE)
# mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary mask

# noise_mask_to_be_removed = denoise_mask(mask)
# # # median_value = statistics.median(var_list)
# # # print("median value",median_value)
# cv2.imwrite("denoised_mask_ca_gilroy_2.jpg",noise_mask_to_be_removed)
# # noise_mask_to_be_removed = cv2.imread("/home/usama/demo161_test_code/edge_detection/noise_mask_to_be_subtracted.jpg",cv2.IMREAD_GRAYSCALE)
# mask_11 = cv2.threshold(noise_mask_to_be_removed, 127, 255, cv2.THRESH_BINARY)[1]
# denoised_mask = cv2.subtract(mask,noise_mask_to_be_removed)
# cv2.imwrite("denoised_mask_6.jpg",denoised_mask)






