import cv2
import os

tile_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/image_tiles_/18_ct_monroe_169.jpg"
mask_tile__path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Cities/ct_monroe_city/output/step6_outputs/mask_tiles_/18_ct_monroe_169.jpg"
original_mask_path = cv2.imread(tile_path)
mask_path = cv2.imread(mask_tile__path)
result = cv2.addWeighted(original_mask_path, 0.5, mask_path, 0.5, 0)
cv2.imwrite("Overlay_mask.png", result)