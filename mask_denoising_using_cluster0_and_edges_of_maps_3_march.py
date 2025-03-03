# import cv2
# import numpy as np

# def apply_adaptive_threshold(image_path, output_path):
#     """
#     Step 1: Perform adaptive thresholding to detect edges.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # Apply adaptive threshold for edge detection
#     edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                   cv2.THRESH_BINARY_INV, 21, 5)
    
#     cv2.imwrite(output_path, edges)
#     return edges


# def combine_mask_with_edges(mask_path, edges, output_path):
#     """
#     Step 2: Combine zoning mask with detected edges.
#     """
#     mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#     # Ensure binary mask
#     _, mask_image = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)

#     # Combine mask and edges using bitwise OR
#     combined_mask = cv2.bitwise_or(mask_image, edges)

#     # Morphological closing to remove small gaps
#     kernel = np.ones((3, 3), np.uint8)
#     combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     cv2.imwrite(output_path, combined_mask)
#     return combined_mask


# def subtract_and_denoise(target_mask_path, combined_mask, output_path):
#     """
#     Step 3: Subtract edges from target cluster mask to denoise.
#     """
#     target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)

#     # Ensure binary mask
#     _, target_mask = cv2.threshold(target_mask, 50, 255, cv2.THRESH_BINARY)

#     # Subtract edges from the mask
#     denoised_mask = cv2.subtract(target_mask, combined_mask)

#     # Apply median blur for smoothing
#     denoised_mask = cv2.medianBlur(denoised_mask, 5)

#     cv2.imwrite(output_path, denoised_mask)
#     return denoised_mask


# def main():
#     # Paths
#     image_path = "/home/usama/test_data_for_sam2_26_feb_2025/input_images/WE3474.jpg"
#     mask_path = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/WE3474-405b Zoning Map-page-001/WE3474-405b Zoning Map-page-001_0.jpg"
#     target_mask_path = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/WE3474-405b Zoning Map-page-001/WE3474-405b Zoning Map-page-001_1.jpg"

#     # Output files
#     edge_output = "we3474_edges.jpg"
#     combined_mask_output = "resultant_mask_image.jpg"
#     denoised_output = "updated_mask.png"

#     # Processing pipeline
#     edges = apply_adaptive_threshold(image_path, edge_output)
#     combined_mask = combine_mask_with_edges(mask_path, edges, combined_mask_output)
#     denoised_mask = subtract_and_denoise(target_mask_path, combined_mask, denoised_output)

#     # Optional visualization
#     cv2.imshow('Edges', edges)
#     cv2.imshow('Combined Mask', combined_mask)
#     cv2.imshow('Denoised Mask', denoised_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()






import cv2
import numpy as np
import os

def find_mask_with_suffix(directory, suffix):
    """
    Find the first file ending with the given suffix in the directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            return os.path.join(directory, filename)
    return None

def apply_adaptive_threshold(image_path, output_path):
    """
    Step 1: Perform adaptive thresholding to detect edges.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply adaptive threshold for edge detection
    edges = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 5)

    cv2.imwrite(output_path, edges)
    return edges

def combine_mask_with_edges(mask_path, edges):
    """
    Step 2: Combine zoning mask with detected edges.
    """
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, mask_image = cv2.threshold(mask_image, 50, 255, cv2.THRESH_BINARY)

    # Combine mask and edges using bitwise OR
    combined_mask = cv2.bitwise_or(mask_image, edges)

    # Morphological closing to remove small gaps
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return combined_mask

def subtract_and_denoise(target_mask_path, combined_mask):
    """
    Step 3: Subtract edges from target cluster mask to denoise.
    """
    target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure binary mask
    _, target_mask = cv2.threshold(target_mask, 50, 255, cv2.THRESH_BINARY)

    # Subtract edges from the mask
    denoised_mask = cv2.subtract(target_mask, combined_mask)

    # Apply median blur for smoothing
    denoised_mask = cv2.medianBlur(denoised_mask, 5)

    return denoised_mask

def process_masks(image_path, mask_directory):
    """
    Main processing loop: find _0.jpg mask, then process all other masks.
    """

    # Ensure output directory exists
    denoised_output_dir = os.path.join(mask_directory, "denoised_masks")
    os.makedirs(denoised_output_dir, exist_ok=True)

    # Find the base _0 mask (this mask remains unchanged and used for all others)
    base_mask_path = find_mask_with_suffix(mask_directory, "_0.jpg")
    if base_mask_path is None:
        print("No _0.jpg mask found in directory.")
        return

    print(f"Using base mask: {base_mask_path}")

    # Detect edges once from the main image
    edge_output = os.path.join(mask_directory, "we3474_edges.jpg")
    edges = apply_adaptive_threshold(image_path, edge_output)

    # Combine base mask with edges (this combined mask will be used for all images)
    combined_mask = combine_mask_with_edges(base_mask_path, edges)

    # Process each mask file (skip _0.jpg itself)
    for filename in os.listdir(mask_directory):
        if filename.endswith(".jpg") and not filename.endswith("_0.jpg"):
            target_mask_path = os.path.join(mask_directory, filename)

            # Apply denoising process
            denoised_mask = subtract_and_denoise(target_mask_path, combined_mask)

            # Save denoised mask with the same name in "denoised_masks" folder
            denoised_output_path = os.path.join(denoised_output_dir, filename)
            cv2.imwrite(denoised_output_path, denoised_mask)
            print(f"Denoised mask saved: {denoised_output_path}")

def main():
    # Paths (example image path, adjust according to your actual directory structure)
    image_path = "/home/usama/test_data_for_sam2_26_feb_2025/input_images/WE3474.jpg"

    # Base directory where masks are stored
    mask_directory = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/WE3474-405b Zoning Map-page-001"

    # Process all masks in directory (first mask is _0, rest are denoised)
    process_masks(image_path, mask_directory)

if __name__ == "__main__":
    main()
