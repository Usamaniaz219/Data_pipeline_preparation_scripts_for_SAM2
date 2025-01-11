"""
The directory structure to run the data pipeline is :
cities:
   {city_name}_city:
      input:
         {city_name}_full_res_masks:
            {city_name}:
               #mask images
               e.g 0.jpg
               1.jpg 
         {city_name}_ori_image:
            #image
            {city_name}.jpg
               
         
      output:
         step1_outputs
            {city_name}_tiles  e.g "ca_dana_point_tiles"
              {city_name}_0.jpg  e.g ca_dana_point_0.jpg
              {city_name}_1.jpg
            
         step2_outputs
            {city_name} # directory
               subdirectory # let say 0,1,2,3
                  suddirectory_tile_0.jpg  e.g 0_tile_0
                  suddirectory_tile_1.jpg
                  suddirectory_tile_2.jpg
               
                subdirectory # let say 0,1,2,3
                  suddirectory_tile_0.jpg  e.g 1_tile_0        # Here the subdirectory contains tiles for masks
                  suddirectory_tile_1.jpg
                  suddirectory_tile_2.jpg

     
         step3_outputs
            subdirectory   let say 0,1,2,3
               {subdirectory}_{city_name}_{number}     e.g  0_fl_indialantic_63.jpg
               {subdirectory}_{city_name}_{number}     e.g  0_fl_indialantic_70.jpg     # Here the subdirectory contains tiles for the original image
               {subdirectory}_{city_name}_{number}     e.g  0_fl_indialantic_78.jpg
            
             subdirectory   let say 0,1,2,3
               {subdirectory}_{city_name}_{number}     e.g  1_fl_indialantic_90.jpg
               {subdirectory}_{city_name}_{number}     e.g  1_fl_indialantic_110.jpg
               {subdirectory}_{city_name}_{number}     e.g  1_fl_indialantic_150.jpg
            

            
         step6_outputs
            image_tiles_
               {subdirectory}_{city_name}_{number}   e.g 0_fl_indialantic_63
            mask_tiles_
               {subdirectory}_{city_name}_{number}   e.g 0_fl_indialantic_63

            # Remember in the step6 we have two folder one contains image tile and the other contains its corresponding mask tile 


"""


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
# input_ori_image_directory = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ca_stanton_city/input/ca_stanton_ori_image/" 
# step1_output = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ca_stanton_city/output/step1_outputs"
input_base_dir = r"C:\Users\VAIO\OneDrive\Desktop\GAN\sam2_cloned_data\input_data\ca_colma/"
image_file_name = os.path.basename(os.path.dirname(input_base_dir))
input_ori_image_dir = input_base_dir+"images"
mask_files_dir = input_base_dir+"masks"
output_base_dir = r"C:\Users\VAIO\OneDrive\Desktop\GAN\sam2_cloned_data\output_data/"
step_1_output = f"{output_base_dir}/{image_file_name}/step_1_outputs/"
step_2_output = f"{output_base_dir}/{image_file_name}/step_2_outputs/"
step_3_output = f"{output_base_dir}/{image_file_name}/step_3_outputs/"
target_image_dir = f"{output_base_dir}/{image_file_name}/step_6_outputs/image_tiles_"
target_mask_dir = f"{output_base_dir}/{image_file_name}/step_6_outputs/mask_tiles_"

os.makedirs(step_1_output, exist_ok=True)
os.makedirs(step_2_output, exist_ok=True)
os.makedirs(step_3_output, exist_ok=True)
os.makedirs(target_image_dir, exist_ok=True)
os.makedirs(target_mask_dir, exist_ok=True)



# # step1:
process_and_save_tiles(input_ori_image_dir, step_1_output, process_image)   # convert the the original image into tiles and then saved them in a directory

# #step2:
process_mask_directory(mask_files_dir, step_2_output) # convert each mask image into tiles and saved them to its corresponding subdirectory

# # step3:
duplicate_images_for_masks(f"{step_1_output}/{image_file_name}_tiles/", f"{step_2_output}/{image_file_name}/", step_3_output)  # duplicates the original image folder and the number of suplicates folders is equal to the number of its masks

# # step4:
clean_up_images_and_masks(step_3_output, f"{step_2_output}/{image_file_name}/") # remove the blank masks tiles and its corresponding image tiles 

# # step5:
rename_images_and_masks(step_3_output,f"{step_2_output}/{image_file_name}/")   # rename the mask tiles based on its corresponding image tiles name

# # step6:
copy_image_and_mask_tiles(step_3_output, f"{step_2_output}/{image_file_name}/",target_image_dir,target_mask_dir) # copy the image tiles and the mask tiles and place the image tiles into the image_tiles_ folder and the mask tiles into the mask_tiles_folder









