import os

# # Define the directories
# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"

# # Iterate over each subdirectory in image_tiles_dir
# for subfolder in os.listdir(image_tiles_dir):
#     image_subfolder_path = os.path.join(image_tiles_dir, subfolder)
#     mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

#     # Ensure corresponding subfolders exist in both directories
#     if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
#         image_files = os.listdir(image_subfolder_path)
#         mask_files = os.listdir(mask_subfolder_path)

#         # Step 1: Rename image files by adding subdirectory name as a prefix
#         new_image_names = {}
#         for image_file in image_files:
#             # Construct the new name with the subfolder as prefix
#             new_image_name = f"{subfolder}_{image_file}"
#             old_image_path = os.path.join(image_subfolder_path, image_file)
#             new_image_path = os.path.join(image_subfolder_path, new_image_name)
            
#             # Rename the image file
#             os.rename(old_image_path, new_image_path)
            
#             # Store the mapping of numeric part to the new image name for mask renaming
#             image_num = image_file.split('_')[-1].split('.')[0]
#             new_image_names[image_num] = new_image_name

#         # Step 2: Rename mask files to match the newly prefixed image file names
#         for mask_file in mask_files:
#             mask_num = mask_file.split('_')[-1].split('.')[0]

#             # Find the corresponding new image name using the numeric part
#             if mask_num in new_image_names:
#                 old_mask_path = os.path.join(mask_subfolder_path, mask_file)
#                 new_mask_path = os.path.join(mask_subfolder_path, new_image_names[mask_num])

#                 # Rename the mask file to match the prefixed image file name
#                 os.rename(old_mask_path, new_mask_path)
#             else:
#                 print(f"Warning: No corresponding image file for {mask_file} in {subfolder}")

# print("Image and mask file renaming completed.")








def rename_images_and_masks(image_tiles_dir, mask_tiles_dir):
    """
    Renames image and mask files in the specified directories by adding the subdirectory name
    as a prefix to each file, ensuring that masks match their corresponding image file names.

    Parameters:
    - image_tiles_dir (str): Directory path containing subdirectories of image tiles.
    - mask_tiles_dir (str): Directory path containing subdirectories of mask tiles.
    """
    
    # Iterate over each subdirectory in image_tiles_dir
    for subfolder in os.listdir(image_tiles_dir):
        image_subfolder_path = os.path.join(image_tiles_dir, subfolder)
        mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

        # Ensure corresponding subfolders exist in both directories
        if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
            image_files = os.listdir(image_subfolder_path)
            mask_files = os.listdir(mask_subfolder_path)

            # Step 1: Rename image files by adding subdirectory name as a prefix
            new_image_names = {}
            for image_file in image_files:
                # Construct the new name with the subfolder as prefix
                new_image_name = f"{subfolder}_{image_file}"
                old_image_path = os.path.join(image_subfolder_path, image_file)
                new_image_path = os.path.join(image_subfolder_path, new_image_name)
                
                # Rename the image file
                os.rename(old_image_path, new_image_path)
                
                # Store the mapping of numeric part to the new image name for mask renaming
                image_num = image_file.split('_')[-1].split('.')[0]
                new_image_names[image_num] = new_image_name

            # Step 2: Rename mask files to match the newly prefixed image file names
            for mask_file in mask_files:
                mask_num = mask_file.split('_')[-1].split('.')[0]

                # Find the corresponding new image name using the numeric part
                if mask_num in new_image_names:
                    old_mask_path = os.path.join(mask_subfolder_path, mask_file)
                    new_mask_path = os.path.join(mask_subfolder_path, new_image_names[mask_num])

                    # Rename the mask file to match the prefixed image file name
                    os.rename(old_mask_path, new_mask_path)
                else:
                    print(f"Warning: No corresponding image file for {mask_file} in {subfolder}")

    print("Image and mask file renaming completed.")



# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"
# # rename_images_and_masks(image_tiles_dir,mask_tiles_dir)





























# import os

# # Define the directories
# image_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/brisbane_old_image_tiles_28_oct_2024/"
# mask_tiles_dir = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/brisbane_old_masks_tiles_28_oct_2024/"

# # Iterate over each subdirectory in image_tiles_dir
# for subfolder in os.listdir(image_tiles_dir):
#     image_subfolder_path = os.path.join(image_tiles_dir, subfolder)
#     mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

#     # Ensure corresponding subfolders exist in both directories
#     if os.path.isdir(image_subfolder_path) and os.path.isdir(mask_subfolder_path):
#         image_files = os.listdir(image_subfolder_path)
#         mask_files = os.listdir(mask_subfolder_path)

#         # Create a dictionary to map image file endings to mask file names
#         image_dict = {}
#         for image_file in image_files:
#             # Extract the unique numeric part for identification
#             image_num = image_file.split('_')[-1].split('.')[0]
#             image_dict[image_num] = image_file

#         for mask_file in mask_files:
#             # Extract the unique numeric part of the mask file
#             mask_num = mask_file.split('_')[-1].split('.')[0]

#             # Find the corresponding image file
#             if mask_num in image_dict:
#                 # Construct the full paths
#                 old_mask_path = os.path.join(mask_subfolder_path, mask_file)
#                 new_mask_path = os.path.join(mask_subfolder_path, image_dict[mask_num])

#                 # Rename the mask file to match the image file name
#                 os.rename(old_mask_path, new_mask_path)
#             else:
#                 print(f"Warning: No corresponding image file for {mask_file} in {subfolder}")
                
# print("Mask file renaming completed.")








