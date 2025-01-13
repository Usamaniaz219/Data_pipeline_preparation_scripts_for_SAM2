# Inference Code #
#######################

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


image_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res/image_tiles_copied/"
mask_path  = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res/mask_tiles_/"

# image_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res/image_tiles_copied/"
# mask_path  = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res/ori_masks_polygons_merged/"

org_mask_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/9_jan_2025_results_meanshift/ca_colma_input/ca_colma_step_6_after_dilation_and_erosion/mask_tiles_copied_11/"  
# txt_files_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/9_jan_2025_results_meanshift/ca_colma_input/ca_colma_step_6_after_dilation_and_erosion/txt_files_3_points/"

def automatic_foreground_prompt_selector_from_image(mask):
     # Binarize the mask
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Apply erosion to ensure points are inside the contours
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # eroded_mask = cv2.erode(mask, kernel, iterations=5)

        # Find contours again on the eroded mask
        # contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare to store selected points for each contour
        all_selected_points = []

        for contour in contours:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter points inside the bounding rectangle but within the contour
            contour_points = [
                (px, py)
                for px in range(x + 1, x + w - 1)  # Exclude the boundary by skipping edges
                for py in range(y + 1, y + h - 1)
                if cv2.pointPolygonTest(contour, (px, py), False) > 0  # Check if inside the contour
            ]
            
            # Randomly select 4 unique points from the filtered points
            if len(contour_points) >= 4:
                if cv2.contourArea(contour)<200:
                    # pass
                    selected_indices = np.random.choice(len(contour_points), 1, replace=False)
                    selected_points = [contour_points[i] for i in selected_indices]
                    print("length of selected points that's area is less than 200",len(selected_points))
                else:
                    selected_indices = np.random.choice(len(contour_points), 3, replace=False)
                    selected_points = [contour_points[i] for i in selected_indices]
                    # print("length of selected points that's area is more than 200",len(selected_points))
            else:
                selected_points = contour_points

            # Add the selected points for the current contour
            all_selected_points.extend(selected_points)
            # print("selected points",selected_points)
        return selected_points,all_selected_points



def automatic_foreground_prompt_selector(mask_images_dir):
    """
    Select points automatically from mask images by processing contours.

    Args:
        mask_images_dir (str): Path to the directory containing mask images.

    Returns:
        dict: A dictionary with filenames as keys and selected points as values.
    """
    selected_points_dict = {}

    # Iterate through all mask images
    for filename in os.listdir(mask_images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read the mask image
            mask_path = os.path.join(mask_images_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            r = np.min([1024 / mask.shape[1], 1024 / mask.shape[0]])  # Scaling factor
            mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)))

            selected_points,all_selected_points = automatic_foreground_prompt_selector_from_image(mask)

            # Store the points in the dictionary
            selected_points_dict[filename] = all_selected_points
            # all_selected_points = all_selected_points.clear()

    return selected_points_dict,selected_points

def load_image_and_points(image_dir, mask_dir,org_mask_dir):
    """
    Load images and corresponding mask points for processing.

    Args:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing mask images.

    Yields:
        tuple: Processed image, resized mask, selected points, and image filename.
    """
    # Get points from the mask using the automatic selector
    points_dict,selected_points = automatic_foreground_prompt_selector(mask_dir)
    # all_selected_points = all_selected_points.clear()
    # print("points dictionary",points_dict)

    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
        print("Processing image:", image_filename)

        # Construct corresponding mask file path
        mask_file_path = os.path.join(mask_dir, image_filename)
        if not os.path.exists(mask_file_path):
            print(f"Mask for {image_filename} not found. Skipping.")
            continue

        # Read image and mask
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Convert BGR to RGB
        mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

        # Get points for the current image
        points = points_dict.get(image_filename, [])
        # points = all_selected_points

        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)))

        # Yield the processed image, mask, points, and filename
        yield Img, mask, np.array(points), image_filename
    



# data_generator_41  = load_image_and_points(image_path,txt_files_path,mask_path)

data_generator_41  = load_image_and_points(image_path,mask_path,org_mask_path)

def calculate_iou(pred_mask, ground_truth_mask):
   
    intersection = np.sum(pred_mask & ground_truth_mask)
    union = np.sum(pred_mask | ground_truth_mask)
    return intersection / union if union > 0 else 0


def testing_loop(input_points):
        

    # for idx,(img1,input_points_21, img_name) in enumerate(data_generator_41):
        # Load the fine-tuned model
        sam2_checkpoint = "sam2_hiera_large.pt"  # @param ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
        model_cfg = "sam2_hiera_l.yaml" 
        # print("input points label",np.ones([input_points_21.shape[0], 1]))
        print("image name",img_name)
        FINE_TUNED_MODEL_WEIGHTS = "fine_tuned_sam2_1_jan_2025_with_8_accumulation_steps_200.torch"
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        # print(sam2_model)
        

        
        # Build net and load weights
        predictor = SAM2ImagePredictor(sam2_model)
        # print("predictor",predictor)
        # print(dir(predictor))
        # predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

        point_label = np.ones([input_points.shape[0], 1])
        point_label = point_label.flatten()
        print("points labels",point_label)

        # Perform inference and predict masks
        with torch.no_grad():
            predictor.set_image(img1)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=point_label)
            
        
        # np_masks = np.array(masks[:, 0])

        # print("np_masks",np_masks.shape)
        if scores.ndim == 1:
            np_masks = np.array(masks)
            np_scores = scores  # If it's 1D, use it directly
            # print("np scores",np_scores)
        else:
            np_masks = np.array(masks[:, 0])
            np_scores = scores[:, 0] 
            # print("np scores",np_scores)
    # print("np scores",np_scores)

        sorted_masks = np_masks[np.argsort(np_scores)][::-1]

        seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)

        mask_11 = sorted_masks
        mask_bool = mask_11.astype(bool)
        # seg_map[mask_bool] = 1  
        
        # cv2.imshow("ground truth",mask_55)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for i in range(mask_bool.shape[0]):
            mask = mask_bool[i]
            seg_map[mask]=1

        seg_map = seg_map.astype(np.uint8)
        _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)
       
        # cv2.imwrite(f"/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/9_jan_2025_results_meanshift/1_jan_checkpoint_results_200_epochs_8_jan_2025_using_original_mask/{img_name}",seg_map)
        # cv2.imwrite(f"/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/9_jan_2025_results_meanshift/1_jan_checkpoint_results_200_epochs_8_jan_2025/_img1_{img_name}",img1)

        # print(idx)
        return seg_map


# Main function
# After

# Ensure the output directory exists
output_txt_files_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res//txt_files_1"
output_masks_dir ="/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/13_jan_2025_automated_inference_res/predicted_masks_1"
os.makedirs(output_txt_files_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

for idx, (img1, gt_mask, input_points_21, img_name) in enumerate(data_generator_41):
    # print("Image shape:", img1.shape)
    # print("Mask shape:", gt_mask.shape)
    print("Points before:", input_points_21)
    print("################################")
    
    # Initial segmentation and IoU
    seg_map = testing_loop(input_points_21)
    iou = calculate_iou(seg_map, gt_mask)
    print("Initial IoU:", iou)

    # Initialize variables to retain the best segmentation map
    best_iou = iou
    best_seg_map = seg_map
    best_prompt = input_points_21  # Initialize with the initial points
    
    max_iterations = 3
    lower_bound = 0.75
    upper_bound = 0.99

    print("Best prompt",best_prompt)

    # Perform refinement if IoU is below the lower bound
    if iou < lower_bound:
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}:")
            print("Image name:", img_name)

            # Generate new points for refinement
            selected_points, all_selected_points_1 = automatic_foreground_prompt_selector_from_image(gt_mask)
            if len(all_selected_points_1) > 0:
                input_points_31 = np.array(all_selected_points_1)
                print("Points after selection:", input_points_31)

                # Generate new segmentation map and compute IoU
                seg_map_ = testing_loop(input_points_31)
                iou = calculate_iou(seg_map_, gt_mask)
                print("Updated IoU:", iou)

                # Update the best segmentation map and IoU if it improves
                if iou > best_iou:
                    best_iou = iou
                    best_seg_map = seg_map_
                    best_prompt = input_points_31  # Update best prompt
                    print("Best IoU updated:", best_iou)

                # Check if IoU is within the acceptable range
                if lower_bound <= iou < upper_bound:
                    print("IoU is now within the acceptable range. Ending process.")
                    break
            else:
                print("No valid points selected. Skipping iteration.")

        # Retain the best segmentation map and IoU after iterations
        print(f"Final retained IoU after refinement: {best_iou}")
    else:
        print(f"Initial IoU {iou} is already within the acceptable range.")

    # Define the file name based on the image name
    # print("image name filename",os.path.splitext(img_name)[0])
    txt_filename = os.path.join(output_txt_files_dir, f"{os.path.splitext(img_name)[0]}.txt")

    
    # Save the best prompt and IoU to the text file
    with open(txt_filename, "w") as file:
       
         for point in best_prompt:
            file.write(f"{point[0]},{point[1]}\n")

    
    cv2.imwrite(f"{output_masks_dir}/{img_name}",best_seg_map)
   
   
    




































# Before

# for idx, (img1, gt_mask, input_points_21, img_name) in enumerate(data_generator_41):
#     print("Image shape",img1.shape)
#     print("mask shape",gt_mask.shape)
#     print("Points before:", input_points_21)
#     print("################################")
    
#     # Initial segmentation and IoU
#     seg_map = testing_loop(input_points_21)
#     iou = calculate_iou(seg_map, gt_mask)
#     print("Initial IoU:", iou)

#     # Initialize variables to retain the best segmentation map
#     best_iou = iou
#     best_seg_map = seg_map
#     max_iterations = 3
#     lower_bound = 0.75
#     upper_bound = 0.99

#     # Perform refinement if IoU is below the lower bound
#     if iou < lower_bound:
#         for iteration in range(max_iterations):
#             print(f"Iteration {iteration + 1}:")
#             print("Image name:", img_name)

#             # Generate new points for refinement
#             selected_points, all_selected_points_1 = automatic_foreground_prompt_selector_from_image(gt_mask)
#             if len(all_selected_points_1) > 0:
#                 input_points_31 = np.array(all_selected_points_1)
#                 print("Points after selection:", input_points_31)

#                 # Generate new segmentation map and compute IoU
#                 seg_map_ = testing_loop(input_points_31)
#                 iou = calculate_iou(seg_map_, gt_mask)
#                 print("Updated IoU:", iou)

#                 # Update the best segmentation map if IoU improves
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_seg_map = seg_map_
#                     print("Best IoU updated:", best_iou)

#                 # Check if IoU is within the acceptable range
#                 if lower_bound <= iou < upper_bound:
#                     print("IoU is now within the acceptable range. Ending process.")
#                     break
#             else:
#                 print("No valid points selected. Skipping iteration.")

#         # Retain the best segmentation map and IoU after iterations
#         print(f"Final retained IoU after refinement: {best_iou}")
#     else:
#         print(f"Initial IoU {iou} is already within the acceptable range.")

#     # Use `best_seg_map` as the final result for this image
#     print("Processing completed for:", img_name)
#     cv2.imshow("best_Seg_map",best_seg_map)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()



































