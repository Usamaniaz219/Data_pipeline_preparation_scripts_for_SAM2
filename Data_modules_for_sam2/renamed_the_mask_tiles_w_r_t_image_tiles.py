

import os

# Define the directories
image_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/brisbane_old_image_tiles_28_oct_2024/"
mask_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/brisbane_old_masks_tiles_28_oct_2024/"

# Iterate over each subdirectory in image_tiles_dir
for subfolder in os.listdir(image_tiles_dir):
    image_subfolder_path = os.path.join(image_tiles_dir, subfolder)
    mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

    # Ensure corresponding subfolders exist in both directories
    if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
        image_files = os.listdir(image_subfolder_path)
        mask_files = os.listdir(mask_subfolder_path)

        # Create a dictionary to map image file endings to mask file names
        image_dict = {}
        for image_file in image_files:
            # Extract the unique numeric part for identification
            image_num = image_file.split('_')[-1].split('.')[0]
            image_dict[image_num] = image_file

        for mask_file in mask_files:
            # Extract the unique numeric part of the mask file
            mask_num = mask_file.split('_')[-1].split('.')[0]

            # Find the corresponding image file
            if mask_num in image_dict:
                # Construct the full paths
                old_mask_path = os.path.join(mask_subfolder_path, mask_file)
                new_mask_path = os.path.join(mask_subfolder_path, image_dict[mask_num])

                # Rename the mask file to match the image file name
                os.rename(old_mask_path, new_mask_path)
            else:
                print(f"Warning: No corresponding image file for {mask_file} in {subfolder}")
                
print("Mask file renaming completed.")








