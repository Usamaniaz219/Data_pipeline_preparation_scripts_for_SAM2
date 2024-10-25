

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


# # Create target directories if they don't exist
# os.makedirs(target_image_dir, exist_ok=True)
# os.makedirs(target_mask_dir, exist_ok=True)

# # Iterate through each subfolder in image_tiles_dir
# for subfolder in os.listdir(image_tiles_dir):
#     image_subfolder_path = os.path.join(image_tiles_dir, subfolder)
#     mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

#     # Check if the corresponding mask subfolder exists
#     if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
#         for image_file in os.listdir(image_subfolder_path):
#             # Extract the numeric part of the image file name for matching
#             image_num = image_file.split('_')[-1].split('.')[0]

#             # Construct the expected mask file name
#             mask_file = f"{subfolder}_tile_{image_num}.jpg"
#             mask_file_path = os.path.join(mask_subfolder_path, mask_file)

#             # If the corresponding mask file exists, copy both files to target folders
#             if os.path.exists(mask_file_path):
#                 # Copy image file
#                 image_file_path = os.path.join(image_subfolder_path, image_file)
#                 shutil.copy(image_file_path, os.path.join(target_image_dir, image_file))

#                 # Copy mask file
#                 shutil.copy(mask_file_path, os.path.join(target_mask_dir, mask_file))
#             else:
#                 print(f"Warning: No corresponding mask file for {image_file} in {subfolder}")

# print("Files copied successfully with one-to-one correspondence.")
