import os
import tiling_the_image_into_800_tiles_and_then_padded_to_make_it_1000
import tiling_each_mask
import copied_the_images_tiles_directory
import Black_tile_and_its_corresponding_tiles_removal
import renamed_the_mask_tiles_w_r_t_image_tiles
import Make_a_paired_image_and_mask_tiles

from tiling_the_image_into_800_tiles_and_then_padded_to_make_it_1000 import process_and_save_tiles,process_image
from tiling_each_mask import process_mask_directory
from copied_the_images_tiles_directory import duplicate_images_for_masks
from Black_tile_and_its_corresponding_tiles_removal import clean_up_images_and_masks
from renamed_the_mask_tiles_w_r_t_image_tiles import rename_images_and_masks
from Make_a_paired_image_and_mask_tiles import copy_image_and_mask_tiles

#(i)
input_ori_image_directory = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/input/ct_monroe_ori_image/" 
step1_output = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step1_outputs"

# step1_tile_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step1_outputs/ct_monroe_tiles/"

name = os.path.basename(os.path.dirname(input_ori_image_directory))
# print(ct_monroe)
city_name = name.split("_ori")[0]
print(city_name)


# (ii)
root_ori_mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/input/ct_monroe_full_res_masks/'  # Root directory containing subdirectories of masks
step2_output = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs'  # Output directory for saving the tiles

# (iii)
# images_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step1_outputs/ct_monroe_tiles/'  # Directory containing the images you want to copy
# step2_masks_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/'  # Directory containing subdirectories for masks
step3_output = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/'  # 

# (iv)
# image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/'
# mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/'

# (v)
# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"

# (vi)
# image_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step3_outputs/"
# mask_tiles_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step2_outputs/ct_monroe/"
target_image_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/image_tiles_"
target_mask_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/mask_tiles_"





# # step1:
process_and_save_tiles(input_ori_image_directory, step1_output, process_image)

# #step2:
process_mask_directory(root_ori_mask_dir, step2_output)

# # step3:
duplicate_images_for_masks(f"{step1_output}/{city_name}_tiles/", f"{step2_output}/{city_name}/", step3_output)

# # step4:
clean_up_images_and_masks(step3_output, f"{step2_output}/{city_name}/")

# # step5:
rename_images_and_masks(step3_output,f"{step2_output}/{city_name}/")

# # step6:
copy_image_and_mask_tiles(step3_output, f"{step2_output}/{city_name}/",target_image_dir,target_mask_dir)









