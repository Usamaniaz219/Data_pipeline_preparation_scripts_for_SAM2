# Inference Code #
#######################

import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from shapely.geometry import Polygon
from shapely.validation import make_valid
import random
import concurrent.futures
import threading

# torch.cuda.memory._record_memory_history(max_entries=100000)

image_path = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/samples_for_time_calculations_19_march_2025/image_tiles/"
dilated_mask_path  = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/samples_for_time_calculations_19_march_2025/mask_tiles/"
org_mask_path ="/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/samples_for_time_calculations_19_march_2025/org_mask_tiles/"



def get_representative_points_within_contours(contours, contours_1,mask):
    """Get representative points within each part of the polygon or a reduced number if there's intersection with contours_1."""
    representative_points = []

    def get_quadrant_representative_points(polygon):
        """Get representative points from the quadrants of a polygon."""
        min_x, min_y, max_x, max_y = polygon.bounds
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        quadrants = [
            Polygon([(min_x, min_y), (center_x, min_y), (center_x, center_y), (min_x, center_y)]),
            Polygon([(center_x, min_y), (max_x, min_y), (max_x, center_y), (center_x, center_y)]),
            Polygon([(min_x, center_y), (center_x, center_y), (center_x, max_y), (min_x, max_y)]),
            Polygon([(center_x, center_y), (max_x, center_y), (max_x, max_y), (center_x, max_y)])
        ]

        temp_points = []  # Temporary list to hold quadrant representative points

        for quadrant in quadrants:
            if quadrant.intersects(polygon):
                intersection = quadrant.intersection(polygon)
                if not intersection.is_empty:
                    rep_point = intersection.representative_point()
                    temp_points.append((rep_point.x, rep_point.y))

        return temp_points
    
    def is_foreground_pixel(x, y, mask):
        """Check if a point lies on the foreground pixel of the annotation mask."""
        rows, cols = mask.shape
        if 0 <= int(y) < rows and 0 <= int(x) < cols:
#             return mask[int(y), int(x)] == 255  # Adjust based on foreground label
            return mask[int(y), int(x)]>0
        return False




    for contour_1 in contours_1:
        try:
            shapely_polygon = Polygon([(point[0][0], point[0][1]) for point in contour_1])
            shapely_polygon = make_valid(shapely_polygon)  # Ensure the polygon is valid
            count = 0
            tmp_pts = []

            for contour in contours:
                # shapely_polygon_1 = Polygon([(point[0][0], point[0][1]) for point in contour])
                coordinates = []
                for cont_point in contour:
                    x = cont_point[0][0]
                    y = cont_point[0][1]
                    coordinates.append((x, y))
                tmp_pts_1 =[]
                if len(coordinates)>3:
                # Create the polygon using the list of coordinates
                    shapely_polygon_1 = Polygon(coordinates)
                    shapely_polygon_1 = make_valid(shapely_polygon_1)  # Ensure the polygon is valid
                    # plot_polygon
                 

                    if shapely_polygon.intersects(shapely_polygon_1):
                        count += 1

                        if shapely_polygon_1.area <= 200:
                            rep_point = shapely_polygon_1.representative_point()
                            representative_points.append(([(rep_point.x, rep_point.y)]))
                            # print("representative point after area is less than 200",representative_points)
                        else:
                            pts = get_quadrant_representative_points(shapely_polygon_1)
                            # print("points11",points)
                            for pt in pts:
                                if is_foreground_pixel(pt[0],pt[1],mask):
                                    tmp_pts_1.append(pt)
                            tmp_pts.append(tmp_pts_1)

                            # tmp_pts.append(get_quadrant_representative_points(shapely_polygon_1))
            if tmp_pts:
                if count > 1:
                    # print("length of tmp_pts",len(tmp_pts))
                    if len(tmp_pts) >= 2:
                        representative_points.append(list(random.sample(tmp_pts[0], 2)))
                        representative_points.append(list(random.sample(tmp_pts[1], 2)))
                    elif tmp_pts:
                        representative_points.append(list(tmp_pts[0]))
                elif count==1:
    #                 rep_point = shapely_polygon.representative_point()
    #                 representative_points.append((rep_point.x, rep_point.y))  # To tackle the case where intersection is not present
                    if tmp_pts:
                    # If no multiple intersections, still get quadrant points
                        
                        representative_points.append(list(tmp_pts[0]))
                        # print(representative_points)
                else:
                    rep_point = shapely_polygon.representative_point()
                    representative_points.append([(rep_point.x, rep_point.y)])  # 

                    # if tmp_pts:
                        
                    # # If no multiple intersections, still get quadrant points
                    #     representative_points.extend(tmp_pts[0])


        except ValueError as e:
            print(f"Error creating polygon: {e}")
            continue

    return representative_points

def process_single_image_using_rep_point_logic(mask):
    """
    Process a single image and its corresponding mask to extract representative points.
    
    Parameters:
        image_path (str): Path to the image file (not used directly in this function, kept for consistency).
        mask_path (str): Path to the mask file.
        output_txt_dir (str, optional): Directory to save the output .txt file with representative points. 
                                        If None, the output is not saved to a file.
    
    Returns:
        rep_points (list): Representative points extracted from the contours of the mask.
    """
    # ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        # print(f"Error: Could not read mask from path {mask_path}")
        return []
    

     # Threshold the mask to create a binary image
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours_1, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Erode the mask and find contours again
    eroded_mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    _, binary_mask_eroded = cv2.threshold(eroded_mask, 127, 255, cv2.THRESH_BINARY)
    contours_2, _ = cv2.findContours(binary_mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Choose the final mask based on contour count
    final_mask = eroded_mask if len(contours_2) >= len(contours_1) else mask
    _, binary_mask_final = cv2.threshold(final_mask, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get representative points with intersection logic
    rep_points = get_representative_points_within_contours(contours, contours_1, mask)
    

    return rep_points

def is_foreground_pixel(x, y, mask):
    """Check if a point lies on the foreground pixel of the annotation mask."""
    rows, cols = mask.shape
    if 0 <= int(y) < rows and 0 <= int(x) < cols:
        # print("mask pixel values",mask[int(y), int(x)])
#             return mask[int(y), int(x)] == 255  # Adjust based on foreground label
        return mask[int(y), int(x)]==255
    # print("is not a foreground point")
    return False


# def automatic_foreground_prompt_selector_from_image(dilated_mask, org_mask, no_of_prompts):
#     # Binarize the mask
#     _, dilated_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)

#     # Find contours in the mask
#     contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Prepare to store selected points for each contour
#     all_selected_points = []

#     for contour in contours:
#         # Calculate the bounding rectangle of the contour
#         x, y, w, h = cv2.boundingRect(contour)

#         # Filter points inside the bounding rectangle but within the contour
#         contour_points = [
#             (px, py)
#             for px in range(x + 2, x + w - 2)  # Exclude the boundary by skipping edges
#             for py in range(y + 2, y + h - 2)
#             if cv2.pointPolygonTest(contour, (px, py), False) > 0  # Check if inside the contour
#         ]
        
#         # Randomly select up to 10 points if enough points exist
#         if len(contour_points) >= 10:
#             if cv2.contourArea(contour) < 200:
#                 selected_indices = np.random.choice(len(contour_points), 1, replace=False)
#             else:
#                 selected_indices = np.random.choice(len(contour_points), 10, replace=False)
#             selected_points = [contour_points[i] for i in selected_indices]
#         else:
#             selected_points = contour_points

#         # Filter only foreground points
#         foreground_points = [pt for pt in selected_points if is_foreground_pixel(pt[0], pt[1], org_mask)]

#         # If all selected points are foreground, randomly pick 4 of them
#         if len(foreground_points)> 4:
#             selected_indices = np.random.choice(len(foreground_points), no_of_prompts, replace=False)
#             foreground_points = [foreground_points[i] for i in selected_indices]
#             all_selected_points.extend(foreground_points)
        
#         else:
#             all_selected_points.extend(foreground_points)

#         # Add the final selected points for the current contour
       
#     return foreground_points, all_selected_points




def automatic_foreground_prompt_selector_from_image(dilated_mask, org_mask, no_of_prompts):
    # Binarize the mask
    _, dilated_mask = cv2.threshold(dilated_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the mask
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare to store selected points for each contour
    all_selected_points = []

    for contour in contours:
        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter points inside the bounding rectangle but within the contour
        contour_points = [
            (px, py)
            for px in range(x + 2, x + w - 2)  # Exclude the boundary by skipping edges
            for py in range(y + 2, y + h - 2)
            if cv2.pointPolygonTest(contour, (px, py), False) > 0  # Check if inside the contour
        ]
        # Filter only foreground points
        foreground_points = [pt for pt in contour_points if is_foreground_pixel(pt[0], pt[1], org_mask)]

        # Randomly select up to 10 points if enough points exist
        if len(foreground_points) > no_of_prompts:
            if cv2.contourArea(contour) < 200:
                selected_indices = np.random.choice(len(foreground_points), 1, replace=False)
            else:
                selected_indices = np.random.choice(len(foreground_points), no_of_prompts, replace=False)
            selected_points = [foreground_points[i] for i in selected_indices]
        else:
            selected_points = foreground_points

        all_selected_points.extend(selected_points)
 
    return selected_points, all_selected_points



def automatic_foreground_prompt_selector_from_directory(dilated_mask_images_dir,org_mask_dir):
    """
    Select points automatically from mask images by processing contours.

    Args:
        mask_images_dir (str): Path to the directory containing mask images.

    Returns:
        dict: A dictionary with filenames as keys and selected points as values.
    """
    selected_points_dict = {}

    # Iterate through all mask images
    for filename in os.listdir(dilated_mask_images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            print("image filename",filename)
            # Read the mask image
            dilated_mask_path = os.path.join(dilated_mask_images_dir, filename)
            org_mask_path = os.path.join(org_mask_dir, filename)
            dilated_mask = cv2.imread(dilated_mask_path, cv2.IMREAD_GRAYSCALE)
            _,dilated_mask = cv2.threshold(dilated_mask,128,255,cv2.THRESH_BINARY)
            org_mask = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)
            
            r = np.min([1024 / dilated_mask.shape[1], 1024 / dilated_mask.shape[0]])  # Scaling factor
            dilated_mask = cv2.resize(dilated_mask, (int(dilated_mask.shape[1] * r), int(dilated_mask.shape[0] * r)))
            org_mask = cv2.resize(org_mask, (int(org_mask.shape[1] * r), int(org_mask.shape[0] * r)))
            # org_mask = cv2.medianBlur(org_mask,5)
            _,org_mask = cv2.threshold(org_mask,128,255,cv2.THRESH_BINARY)
            # cv2.imshow("org_mask.jpg",org_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            selected_points,all_selected_points = automatic_foreground_prompt_selector_from_image(dilated_mask,org_mask, 4)
            # print("all selected points",all_selected_points)

            # Store the points in the dictionary
            selected_points_dict[filename] = all_selected_points
            # all_selected_points = all_selected_points.clear()

    return selected_points_dict,selected_points

def load_image_and_points(image_dir, dilated_mask_dir,org_mask_dir):
    """
    Load images and corresponding mask points for processing.

    Args:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing mask images.

    Yields:
        tuple: Processed image, resized mask, selected points, and image filename.
    """
    # Get points from the mask using the automatic selector
    points_dict,selected_points = automatic_foreground_prompt_selector_from_directory(dilated_mask_dir,org_mask_dir)

    mask_files = [f for f in os.listdir(dilated_mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in mask_files:
        print("Processing image:", image_filename)

        # Construct corresponding mask file path
        dilated_mask_file_path = os.path.join(dilated_mask_dir, image_filename)
        org_mask_path = os.path.join(org_mask_dir,image_filename)
        if not os.path.exists(dilated_mask_path):
            print(f"Mask for {image_filename} not found. Skipping.")
            continue

        # Read image and mask
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Convert BGR to RGB
        gt_dilated_mask = cv2.imread(dilated_mask_file_path, cv2.IMREAD_GRAYSCALE)
        gt_org_mask = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)

        # Get points for the current image
        points = points_dict.get(image_filename, [])
        points_after = np.array(points).reshape(-1, 1, 2)
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        gt_dilated_mask = cv2.resize(gt_dilated_mask, (int(gt_dilated_mask.shape[1] * r), int(gt_dilated_mask.shape[0] * r)))
        gt_org_mask = cv2.resize(gt_org_mask, (int(gt_org_mask.shape[1] * r), int(gt_org_mask.shape[0] * r)))

        # Yield the processed image, mask, points, and filename
        yield Img, gt_dilated_mask,gt_org_mask, points_after, image_filename
    

# data_generator_41  = load_image_and_points(image_path,txt_files_path,mask_path)

data_generator_41  = load_image_and_points(image_path,dilated_mask_path,org_mask_path)

#models = [
 #   ("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_tiny_23_march_2025_8_accumulations_step_55.torch","tiny"),
  #  ("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_large.pt", "sam2_hiera_l.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_13_feb_2025_with_8_accumulation_steps_90.torch","large")
#]


models = [("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_large.pt", "sam2_hiera_l.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_13_feb_2025_with_8_accumulation_steps_90.torch","large"),
("/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml", "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/17_feb_2025_checkpoints/fine_tuned_sam2_tiny_23_march_2025_8_accumulations_step_55.torch","tiny")
]


def load_model(model_cfg,sam2_checkpoint,fine_tuned_sam2_checkpoint):
    sam2_model_cuda = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    sam2_model_cpu = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    # Build net and load weights
    predictor_cuda = SAM2ImagePredictor(sam2_model_cuda)
    predictor_cpu = SAM2ImagePredictor(sam2_model_cpu)
    predictor_cuda.model.load_state_dict(torch.load(fine_tuned_sam2_checkpoint))
    predictor_cpu.model.load_state_dict(torch.load(fine_tuned_sam2_checkpoint, map_location="cpu"))
    # predictor = predictor_cuda
    return predictor_cuda,predictor_cpu



def calculate_iou(pred_mask, ground_truth_mask):
   
    intersection = np.sum(pred_mask & ground_truth_mask)
    union = np.sum(pred_mask | ground_truth_mask)
    return intersection / union if union > 0 else 0


def testing_loop(input_points):
        
    point_label = np.ones([input_points.shape[0], 1])
      
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=point_label)
        
    if scores.ndim == 1:
        np_masks = np.array(masks)
        np_scores = scores  # If it's 1D, use it directly
    else:
        np_masks = np.array(masks[:, 0])
        np_scores = scores[:, 0] 

    sorted_masks = np_masks[np.argsort(np_scores)][::-1]

    seg_map = np.zeros_like(sorted_masks[0], dtype=np.uint8)

    mask_11 = sorted_masks
    mask_bool = mask_11.astype(bool)

    for i in range(mask_bool.shape[0]):
        mask = mask_bool[i]
        seg_map[mask]=1

    seg_map = seg_map.astype(np.uint8)
    _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)
    
    # print(idx)
    return seg_map

max_iterations = 15
lower_bound = 0.99
upper_bound = 1.0
semaphore = threading.BoundedSemaphore(4)

def calculate_best_iou(gt_dilated_mask, gt_org_mask, no_of_prompt, best_iou, best_seg_map, best_prompt):
    # with semaphore:
        for iteration in range(max_iterations):
            print(f"{no_of_prompt} point selection Iteration {iteration + 1}:")
            
            # Generate new points for refinement
            selected_points, all_selected_points_1 = automatic_foreground_prompt_selector_from_image(gt_dilated_mask,gt_org_mask, no_of_prompt)
            if len(all_selected_points_1) > 0:
                input_points_31 =  np.array(all_selected_points_1).reshape(-1, 1, 2)
                # Generate new segmentation map and compute IoU
                seg_map_ = testing_loop(input_points_31)
                # iou = calculate_iou(seg_map_, gt_dilated_mask)
                iou = calculate_iou(seg_map_, gt_org_mask)
                # print("Updated IoU:", iou)

                # Update the best segmentation map and IoU if it improves
                if iou > best_iou:
                    best_iou = iou
                    best_seg_map = seg_map_
                    best_prompt = input_points_31  # Update best prompt
                    # print("Best IoU updated:", best_iou)

                # Check if IoU is within the acceptable range
                if lower_bound <= iou < upper_bound:
                    print("IoU is now within the acceptable range. Ending process.")
                    break
        return best_iou, best_seg_map, best_prompt


# Ensure the output directory exists
output_txt_files_dir = "/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/txt_files_26_march_2025_33"
output_masks_dir ="/media/usama/SSD/Usama_dev_ssd/Zoning_segmentation_code/image_segmentation_and_denoising/clustering_results_26_feb_2025/4_maps_results/demo141/test_Samples_12_march_2025/sam2_outputs_with_no_parrallel_processing/mask_files_26_march_2025_33"
# output_images_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Evaluation_results_17_feb_2025/image_files_21_feb_2025_30_itterations_test_set_25_feb_2025_latest_with_one_two_three_and_four_points"

os.makedirs(output_txt_files_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)
# os.makedirs(output_images_dir, exist_ok=True)




for idx, (img1, gt_dilated_mask,gt_org_mask, input_points_21, img_name) in enumerate(data_generator_41):
    # edged = cv2.Canny(gt_dilated_mask, 50, 150) 
    gt_dilated_mask_thresholded = cv2.threshold(gt_dilated_mask,128,255,cv2.THRESH_BINARY)[1]

 
    best_iou, best_seg_map = 0, None
    best_prompt = input_points_21

    for sam2_checkpoint, model_cfg, finetuned_sam2_checkpoint,model_name in models:
        print(f"{model_name} is loaded!")
        predictor_cuda,predictor_cpu = load_model(model_cfg, sam2_checkpoint,finetuned_sam2_checkpoint)
        contours, _ = cv2.findContours(gt_dilated_mask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        no_of_polygons = len(contours)

        total_no_of_prompts = (4 * no_of_polygons) +( 3 * no_of_polygons) + (2 * no_of_polygons) + (no_of_polygons)
        if total_no_of_prompts > 100:
            print(f"no of prompts is greater than 100 so computation is run on cpu.total no of prompts is {total_no_of_prompts}")  
            predictor = predictor_cpu
        else:
            predictor = predictor_cuda
        with torch.no_grad():
            predictor.set_image(img1)
            print("################################")
            # print("Image name:", img_name)

            
            # Initial segmentation and IoU
            seg_map = testing_loop(input_points_21)
            # iou = calculate_iou(seg_map, gt_dilated_mask)
            iou = calculate_iou(seg_map, gt_org_mask)
            # print("Initial IoU:", iou)

            # # Initialize variables to retain the best segmentation map
            # best_iou = iou
            # best_seg_map = seg_map
            # best_prompt = input_points_21  # Initialize with the initial points
            
            # lower_bound = 0.88
            

            _,gt_org_mask = cv2.threshold(gt_org_mask,128,255,cv2.THRESH_BINARY)
        
            _,gt_dilated_mask = cv2.threshold(gt_dilated_mask,128,255,cv2.THRESH_BINARY)

            # Perform refinement if IoU is below the lower bound
            if iou < lower_bound:
                prompt_list = [1,2,3,4]
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Start the load operations and mark each future with its URL
                    for promp in prompt_list:
                        futures.append(executor.submit(calculate_best_iou, gt_dilated_mask, gt_org_mask, promp, best_iou, best_seg_map, best_prompt))
                    for future in concurrent.futures.as_completed(futures):
                        current_iou, current_seg_map, current_prompt = future.result()
                        # print(f"current iou {model_name}",current_iou)
                        if current_iou > best_iou:
                            best_iou = current_iou
                            best_seg_map = current_seg_map
                            best_prompt = current_prompt

                    

         
        print("###########################################################################")
        # print("before")
        print(f"Best Iou {model_name} :", best_iou)
        print("###########################################################################")
    
    if best_iou < lower_bound:
            # Process the image using representative point logic
            rep_points = process_single_image_using_rep_point_logic(gt_dilated_mask)
        
            print("Rep points:", rep_points)
            
            # Initialize transformed group
            transformed_group = []
            
            
            for rep_group in rep_points:
                for coord in rep_group:
                    # Ensure the coordinate is checked for being on the foreground
                    x, y = coord
                    if is_foreground_pixel(x, y, gt_org_mask):
                        # Append as a single-item nested list
                        transformed_group.append([[int(x), int(y)]])
                    else:
                        print(f"Point ({x}, {y}) is not on the foreground.")
            
            # Convert the list to a numpy array
            if len(transformed_group)>0:
                rep_array = np.array(transformed_group, dtype=np.int32)
                # print("Points after rep point logic and foreground check:", rep_array)
            
                # Process the transformed points
                
                seg_map_ = testing_loop(rep_array)
                print("Ground truth mask shape:", gt_dilated_mask.shape)
                iou = calculate_iou(seg_map_, gt_dilated_mask)
                print("Updated IoU after representative point:", iou)
            
                if iou > best_iou:
                    best_iou = iou
                    best_seg_map = seg_map_
                    best_prompt = rep_array  # Update best prompt
                    print(f"Best IoU updated of {model_name}:", best_iou)




    txt_filename = os.path.join(output_txt_files_dir, f"{os.path.splitext(img_name)[0]}.txt")

    
    # # Save the best prompt and IoU to the text file
    with open(txt_filename, "w") as file:
    
        for point in best_prompt:
            for pt in point:
                # if len(point) == 2:
                file.write(f"{pt[0]},{pt[1]}\n")
    
    
    # if best_iou < 0.96:
    #     print(f"Best Iou {best_iou} is less than 0.96")
    #     best_seg_map = gt_org_mask.copy()
    #     print("Dilated image is returned!")
    
    cv2.imwrite(f"{output_masks_dir}/{img_name}",best_seg_map)
   