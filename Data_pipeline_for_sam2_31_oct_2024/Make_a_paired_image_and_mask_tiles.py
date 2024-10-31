# import os
# import shutil

# # Define source and target directories
# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"
# target_image_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/image_tiles_"
# target_mask_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/mask_tiles_"

# # Create target directories if they don't exist
# os.makedirs(target_image_dir, exist_ok=True)
# os.makedirs(target_mask_dir, exist_ok=True)

# # Move files from each subdirectory in image_tiles_dir to target_image_dir
# for subfolder in os.listdir(image_tiles_dir):
#     image_subfolder_path = os.path.join(image_tiles_dir, subfolder)

#     if os.path.isdir(image_subfolder_path):
#         for image_file in os.listdir(image_subfolder_path):
#             # Construct full file path and move the file
#             source_image_file = os.path.join(image_subfolder_path, image_file)
#             target_image_file = os.path.join(target_image_dir, image_file)
#             shutil.copy(source_image_file, target_image_file)

# # Move files from each subdirectory in mask_tiles_dir to target_mask_dir
# for subfolder in os.listdir(mask_tiles_dir):
#     mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

#     if os.path.isdir(mask_subfolder_path):
#         for mask_file in os.listdir(mask_subfolder_path):
#             # Construct full file path and move the file
#             source_mask_file = os.path.join(mask_subfolder_path, mask_file)
#             target_mask_file = os.path.join(target_mask_dir, mask_file)
#             shutil.copy(source_mask_file, target_mask_file)

# print("Files copied successfully to image_tiles_25_oct_latest and mask_tiles_25_oct_latest.")






import os
import shutil

def copy_image_and_mask_tiles(image_tiles_dir, mask_tiles_dir, target_image_dir, target_mask_dir):
    """
    Copies image and mask files from subdirectories in the source directories to target directories.
    
    Parameters:
    - image_tiles_dir (str): Source directory containing subdirectories of image tiles.
    - mask_tiles_dir (str): Source directory containing subdirectories of mask tiles.
    - target_image_dir (str): Target directory for copied image files.
    - target_mask_dir (str): Target directory for copied mask files.
    """
    
    # Create target directories if they don't exist
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    # Move files from each subdirectory in image_tiles_dir to target_image_dir
    for subfolder in os.listdir(image_tiles_dir):
        image_subfolder_path = os.path.join(image_tiles_dir, subfolder)

        if os.path.isdir(image_subfolder_path):
            for image_file in os.listdir(image_subfolder_path):
                # Construct full file path and move the file
                source_image_file = os.path.join(image_subfolder_path, image_file)
                target_image_file = os.path.join(target_image_dir, image_file)
                shutil.copy(source_image_file, target_image_file)

    # Move files from each subdirectory in mask_tiles_dir to target_mask_dir
    for subfolder in os.listdir(mask_tiles_dir):
        mask_subfolder_path = os.path.join(mask_tiles_dir, subfolder)

        if os.path.isdir(mask_subfolder_path):
            for mask_file in os.listdir(mask_subfolder_path):
                # Construct full file path and move the file
                source_mask_file = os.path.join(mask_subfolder_path, mask_file)
                target_mask_file = os.path.join(target_mask_dir, mask_file)
                shutil.copy(source_mask_file, target_mask_file)

    print("Files copied successfully to target directories.")


# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"
# target_image_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/image_tiles_"
# target_mask_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/mask_tiles_"


# copy_image_and_mask_tiles(image_tiles_dir, mask_tiles_dir,target_image_dir,target_mask_dir)
