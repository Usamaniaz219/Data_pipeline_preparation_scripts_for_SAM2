

import os

# Define the directories
image_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs_24_oct_latest/image_tiles_dir/"
mask_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs_24_oct_latest/masks_tiles_dir/"

# Iterate over each subdirectory in mask_tiles_dir
for subfolder in os.listdir(mask_tiles_dir):
    mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)
    image_subfolder_path = os.path.join(image_tiles_dir, subfolder)

    # Ensure corresponding subfolders exist in both directories
    if os.path.isdir(mask_subfolder_path) and os.path.isdir(image_subfolder_path):
        mask_files = os.listdir(mask_subfolder_path)
        image_files = os.listdir(image_subfolder_path)

        # Create a dictionary to map mask file endings to image file names
        mask_dict = {}
        for mask_file in mask_files:
            # Extract the unique numeric part for identification
            mask_num = mask_file.split('_')[-1].split('.')[0]
            mask_dict[mask_num] = mask_file

        for image_file in image_files:
            # Extract the unique numeric part of the image file
            image_num = image_file.split('_')[-1].split('.')[0]

            # Find the corresponding mask file
            if image_num in mask_dict:
                # Construct the full paths
                old_image_path = os.path.join(image_subfolder_path, image_file)
                new_image_path = os.path.join(image_subfolder_path, mask_dict[image_num])

                # Rename the image file to match the mask file name
                os.rename(old_image_path, new_image_path)
            else:
                print(f"Warning: No corresponding mask file for {image_file} in {subfolder}")
                
print("Image file renaming completed.")


