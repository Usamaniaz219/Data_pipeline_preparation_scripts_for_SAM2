import os
import shutil

def duplicate_images_for_masks(images_dir, masks_dir, output_dir):
    # Step 1: Get the list of subdirectories inside the masks directory
    mask_subfolders = [subdir for subdir in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, subdir))]
    
    print(f"Found {len(mask_subfolders)} subdirectories in the masks directory.")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Iterate over each subfolder in the masks directory
    for subfolder in mask_subfolders:
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)  # Create the subdirectory if it doesn't exist

        # Step 3: Copy all files from the images directory to the new subfolder
        for image_file in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_file)
            if os.path.isfile(image_path):  # Ensure it's a file, not a directory
                shutil.copy(image_path, output_subfolder)  # Copy the image to the new subfolder
                print(f"Copied {image_file} to {output_subfolder}")

# images_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step1_outputs/ct_monroe_tiles/'  # Directory containing the images you want to copy
# masks_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/'  # Directory containing subdirectories for masks
# output_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs'  # Directory where the new subfolders with images should be created
# # images_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/image_tiles_/'
# # masks_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/tiling_masks_outputs_11/'
# # output_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/copied_image_tiles/'

# duplicate_images_for_masks(images_dir, masks_dir, output_dir)


