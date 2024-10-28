import os
import shutil

def duplicate_images_for_masks(images_dir, masks_dir, output_dir):
    # Step 1: Get the list of subdirectories inside the masks directory
    mask_subfolders = [subdir for subdir in os.listdir(masks_dir) if os.path.isdir(os.path.join(masks_dir, subdir))]
    
    print(f"Found {len(mask_subfolders)} subdirectories in the masks directory.")
    
    # Step 2: Iterate over each subfolder in the masks directory
    for subfolder in mask_subfolders:
        # Create a corresponding subdirectory in the output directory
        output_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(output_subfolder, exist_ok=True)  # Create the subdirectory if it doesn't exist

        # Step 3: Copy all files from the images directory to the new subfolder
        for image_file in os.listdir(images_dir):
            image_path = os.path.join(images_dir, image_file)
            if os.path.isfile(image_path):  # Ensure it's a file, not a directory
                shutil.copy(image_path, output_subfolder)  # Copy the image to the new subfolder
                print(f"Copied {image_file} to {output_subfolder}")

images_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs_24_oct_latest_results/Clewiston-Zoning-Map-page-001_modified_image_tiles/'  # Directory containing the images you want to copy
masks_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/tiling_masks_outputs/Clewiston-Zoning-Map-page-001_modified_masks/'  # Directory containing subdirectories for masks
output_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/tiling_images_and_masks_tiles_outputs/image_tiles_dir/'  # Directory where the new subfolders with images should be created

duplicate_images_for_masks(images_dir, masks_dir, output_dir)
