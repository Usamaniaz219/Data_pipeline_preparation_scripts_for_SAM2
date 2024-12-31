
import os
import shutil

# def move_selected_masks(mask_dir, txt_file, output_dir):
 
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Read mask names from the text file
#     with open(txt_file, 'r') as file:
#         mask_names = file.read().splitlines()

#     # Iterate over the mask names and move them if they exist in the mask directory
#     for mask_name in mask_names:
#         mask_name = mask_name+".jpg"
#         mask_path = os.path.join(mask_dir, mask_name)
#         print(mask_path)
#         # mask_path = mask_path + ".jpg"
#         if os.path.exists(mask_path):
#             shutil.move(mask_path, os.path.join(output_dir, mask_name))
#             print(f"Moved: {mask_name}")
#         else:
#             print(f"File not found: {mask_name}")

# mask_dir = "/home/usama/wajeeha_dataset_2/sam2_zone_based_dataset_Final/masks/"
# txt_file = "/home/usama/wajeeha_dataset_2/sam2_zone_dataset_final.txt"
# output_dir = "/home/usama/wajeeha_dataset_2/sam2_zone_based_dataset_Final_small_polygons_to_remove/"

# move_selected_masks(mask_dir, txt_file, output_dir)





def move_selected_txts(txt_dir, txt_file, output_dir):
 
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read mask names from the text file
    with open(txt_file, 'r') as file:
        mask_names = file.read().splitlines()

    # Iterate over the mask names and move them if they exist in the mask directory
    for mask_name in mask_names:
        mask_name = mask_name+".jpg"
        mask_path = os.path.join(txt_dir, mask_name)
        print(mask_path)
        # mask_path = mask_path + ".txt"
        if os.path.exists(mask_path):
            shutil.move(mask_path, os.path.join(output_dir, mask_name))
            print(f"Moved: {mask_name}")
        else:
            print(f"File not found: {mask_name}")

txt_dir = "/home/usama/wajeeha_dataset_2/sam2_zone_based_dataset_Final/images/"
txt_file = "/home/usama/wajeeha_dataset_2/sam2_zone_dataset_final.txt"
output_dir = "/home/usama/wajeeha_dataset_2/sam2_zone_based_dataset_Final_images_small_polygons_to_remove/"

move_selected_txts(txt_dir, txt_file, output_dir)