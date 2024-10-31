import os
import cv2
import re
import numpy as np

def is_black_image(image_path):
    """
    Check if an image is completely black (i.e., all pixel values are zero).
    Returns True if the image is black, otherwise False.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return False
    return np.count_nonzero(image) == 0

def clean_up_images_and_masks(image_dir, mask_dir):
    # Step 1: Get the list of subdirectories
    image_subfolders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    mask_subfolders = [f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))]

    # Step 2: Process each corresponding subdirectory
    for subfolder in image_subfolders:
        image_subfolder_path = os.path.join(image_dir, subfolder)
        mask_subfolder_path = os.path.join(mask_dir, subfolder)

        # Check if the corresponding mask subfolder exists
        if not os.path.exists(mask_subfolder_path):
            print(f"Error: Missing corresponding mask subfolder for {subfolder}")
            continue

        # Step 3: Iterate through files in the image subfolder
        for image_file in os.listdir(image_subfolder_path):
            image_file_path = os.path.join(image_subfolder_path, image_file)

            # Extract the numerical ID from the image filename
            image_id_match = re.search(r'_(\d+)\.jpg$', image_file)
            if not image_id_match:
                print(f"Warning: Skipping file with unexpected name format: {image_file}")
                continue
            image_id = image_id_match.group(1)

            # Construct the expected mask filename
            mask_file = f"{subfolder}_tile_{image_id}.jpg"
            mask_file_path = os.path.join(mask_subfolder_path, mask_file)

            # Check if the mask file exists
            if not os.path.exists(mask_file_path):
                print(f"Error: Missing corresponding mask file for {image_file}")
                continue

            # Step 4: Check if the mask is black
            if is_black_image(mask_file_path):
                # If the mask is black, delete both mask and corresponding image tile
                os.remove(mask_file_path)
                os.remove(image_file_path)
                # print(f"Deleted black mask and corresponding image: {mask_file} and {image_file}")



# image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/'
# mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/'
# clean_up_images_and_masks(image_dir, mask_dir)





























# import os
# import cv2
# import numpy as np

# def is_black_image(image_path):
#     """
#     Check if an image is completely black (i.e., all pixel values are zero).
#     Returns True if the image is black, otherwise False.
#     """
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
#     if image is None:
#         print(f"Error: Unable to load image {image_path}")
#         return False
#     # Check if all pixel values are zero (black)
#     return np.count_nonzero(image) == 0

# def clean_up_images_and_masks(image_dir, mask_dir):
#     # Step 1: Get the list of subdirectories
#     image_subfolders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
#     mask_subfolders = [f for f in os.listdir(mask_dir) if os.path.isdir(os.path.join(mask_dir, f))]

#     # Step 2: Ensure both directories have the same number of subfolders
#     if len(image_subfolders) != len(mask_subfolders):
#         print("Error: Mismatch in the number of subdirectories between images and masks.")
#         return

#     # Step 3: Process each corresponding subdirectory
#     for subfolder in image_subfolders:
#         image_subfolder_path = os.path.join(image_dir, subfolder)
#         mask_subfolder_path = os.path.join(mask_dir, subfolder)

#         if not os.path.exists(mask_subfolder_path):
#             print(f"Error: Missing corresponding mask subfolder for {subfolder}")
#             continue

#         # Step 4: Iterate through files in the subfolder
#         for mask_file in os.listdir(mask_subfolder_path):
#             mask_file_path = os.path.join(mask_subfolder_path, mask_file)

#             # Ensure the corresponding image tile exists
#             image_file_path = os.path.join(image_subfolder_path, mask_file)
#             if not os.path.exists(image_file_path):
#                 print(f"Error: Missing corresponding image file for {mask_file}")
#                 continue

#             # Step 5: Check if the mask is black
#             if is_black_image(mask_file_path):
#                 # If the mask is black, delete both mask and corresponding image tile
#                 os.remove(mask_file_path)
#                 os.remove(image_file_path)
#                 print(f"Deleted black mask and corresponding image: {mask_file}")

# # Example usage
# image_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/Clewiston-Zoning-Map-page-001_modified_tile_to_tile_image_and_mask_data_24_oct_2024/image_tiles_dir/'  # Directory containing image subfolders
# mask_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/Clewiston-Zoning-Map-page-001_modified_tile_to_tile_image_and_mask_data_24_oct_2024/mask_tiles_dir/'  # Directory containing mask subfolders
# clean_up_images_and_masks(image_dir, mask_dir)
