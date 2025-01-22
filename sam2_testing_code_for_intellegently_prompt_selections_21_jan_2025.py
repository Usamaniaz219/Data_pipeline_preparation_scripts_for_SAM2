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



image_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/16_jan_2025_automated_inference_res/demo161_17_jan_2025_data/demo_161_/images/"
dilated_mask_path  = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/16_jan_2025_automated_inference_res/demo161_17_jan_2025_data/demo_161_/masks/"
org_mask_path = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Testing_SAM2_on_Meanshift_data/demo161/outputs/demo161/step_6_outputs_merged_res/mask_tiles_org/"

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

            if count > 1:
                print("length of tmp_pts",len(tmp_pts))
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
#             return mask[int(y), int(x)] == 255  # Adjust based on foreground label
        return mask[int(y), int(x)]==255
    # print("is not a foreground point")
    return False



def automatic_foreground_prompt_selector_from_image(dilated_mask,org_mask):
     # Binarize the mask
    
        _, dilated_mask = cv2.threshold(dilated_mask, 20, 255, cv2.THRESH_BINARY)

        # cv2.imshow("dilated mask",dilated_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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
            
            # Randomly select 4 unique points from the filtered points
            if len(contour_points) >= 4:
                if cv2.contourArea(contour)<200:
                    # pass
                    selected_indices = np.random.choice(len(contour_points), 1, replace=False)
                    selected_points = [contour_points[i] for i in selected_indices]
                    # print("length of selected points that's area is less than 200",len(selected_points))
                else:
                    selected_indices = np.random.choice(len(contour_points), 4, replace=False)
                    selected_points = [contour_points[i] for i in selected_indices]
                    # print("length of selected points that's area is more than 200",len(selected_points))
            else:
                selected_points = contour_points

            for sel_pt in selected_points:
                if is_foreground_pixel(sel_pt[0],sel_pt[1],org_mask):
                    all_selected_points.append(sel_pt)
                else:
                    print("selected point is not foreground")


                # print("selected points",sel_pt)

            # Add the selected points for the current contour
            # all_selected_points.extend(selected_points)
            # print("selected points",selected_points)
        return selected_points,all_selected_points



def automatic_foreground_prompt_selector_from_directory(dilated_mask_images_dir,org_mask_dir):
    """
    Select points automatically from mask images by processing contours.

    Args:
        mask_images_dir (str): Path to the directory containing mask images.

    Returns:
        dict: A dictionary with filenames as keys and selected points as values.
    """
    selected_points_dict = {}

    dilated_mask_files = [f for f in os.listdir(dilated_mask_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    org_mask_files = [f for f in os.listdir(org_mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate through all mask images
    for filename in os.listdir(dilated_mask_images_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Read the mask image
            dilated_mask_path = os.path.join(dilated_mask_images_dir, filename)
            org_mask_path = os.path.join(org_mask_dir, filename)
            dilated_mask = cv2.imread(dilated_mask_path, cv2.IMREAD_GRAYSCALE)
            # _,dilated_mask = cv2.threshold(dilated_mask,20,255,cv2.THRESH_BINARY)
            org_mask = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)
            _,org_mask = cv2.threshold(org_mask,20,255,cv2.THRESH_BINARY)
            r = np.min([1024 / dilated_mask.shape[1], 1024 / dilated_mask.shape[0]])  # Scaling factor
            dilated_mask = cv2.resize(dilated_mask, (int(dilated_mask.shape[1] * r), int(dilated_mask.shape[0] * r)))
            org_mask = cv2.resize(org_mask, (int(org_mask.shape[1] * r), int(org_mask.shape[0] * r)))

            selected_points,all_selected_points = automatic_foreground_prompt_selector_from_image(dilated_mask,org_mask)
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
    # all_selected_points = all_selected_points.clear()
    # print("points dictionary",points_dict)

    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
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
        # print("points before",points)
        # points = all_selected_points
        points_after = np.array(points).reshape(-1, 1, 2)
        # print("points after",points_after)
        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        gt_dilated_mask = cv2.resize(gt_dilated_mask, (int(gt_dilated_mask.shape[1] * r), int(gt_dilated_mask.shape[0] * r)))
        gt_org_mask = cv2.resize(gt_org_mask, (int(gt_org_mask.shape[1] * r), int(gt_org_mask.shape[0] * r)))

        # Yield the processed image, mask, points, and filename
        yield Img, gt_dilated_mask,gt_org_mask, points_after, image_filename
    



# data_generator_41  = load_image_and_points(image_path,txt_files_path,mask_path)

data_generator_41  = load_image_and_points(image_path,dilated_mask_path,org_mask_path)

def calculate_iou(pred_mask, ground_truth_mask):
   
    intersection = np.sum(pred_mask & ground_truth_mask)
    union = np.sum(pred_mask | ground_truth_mask)
    return intersection / union if union > 0 else 0


def testing_loop(input_points):
        
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
        predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))
        
        point_label = np.ones([input_points.shape[0], 1])
        # point_label = point_label.flatten()
        # print("points labels",point_label)

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
        print("Mask bool shape",mask_bool.shape)
        # seg_map[mask_bool] = 1  
        
        # cv2.imshow("ground truth",mask_55)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        for i in range(mask_bool.shape[0]):
            mask = mask_bool[i]
            seg_map[mask]=1

        seg_map = seg_map.astype(np.uint8)
        _,seg_map = cv2.threshold(seg_map,0,255,cv2.THRESH_BINARY)
       
        # print(idx)
        return seg_map


# Main function
# After

# Ensure the output directory exists
output_txt_files_dir = "/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/21_jan_2025_automated_inference_res/demo161_failed_txt_files_2_iter_lower_bound_increased"
output_masks_dir ="/media/usama/SSD/Data_for_SAM2_model_Finetuning/Sam2_fine_tuning_/segment-anything-2/21_jan_2025_automated_inference_res/demo161_predicted_masks_failed_2_iter_lower_bound_increased"
os.makedirs(output_txt_files_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

for idx, (img1, gt_dilated_mask,gt_org_mask, input_points_21, img_name) in enumerate(data_generator_41):
    # print("Image shape:", img1.shape)
    # print("Mask shape:", gt_mask.shape)
    # print("Points before:", input_points_21)
    print("################################")

    
    # Initial segmentation and IoU
    seg_map = testing_loop(input_points_21)
    iou = calculate_iou(seg_map, gt_dilated_mask)
    print("Initial IoU:", iou)

    # Initialize variables to retain the best segmentation map
    best_iou = iou
    best_seg_map = seg_map
    best_prompt = input_points_21  # Initialize with the initial points
    
    max_iterations = 2
    # lower_bound = 0.88
    lower_bound = 0.96
    upper_bound = 0.99

    _,gt_org_mask = cv2.threshold(gt_org_mask,20,255,cv2.THRESH_BINARY)
    _,gt_dilated_mask = cv2.threshold(gt_dilated_mask,20,255,cv2.THRESH_BINARY)

    print("Best prompt",best_prompt)

    # Perform refinement if IoU is below the lower bound
    if iou < lower_bound:
        for iteration in range(max_iterations):
            print(f"Iteration {iteration + 1}:")
            print("Image name:", img_name)
            
            # Generate new points for refinement
            selected_points, all_selected_points_1 = automatic_foreground_prompt_selector_from_image(gt_dilated_mask,gt_org_mask)
            if len(all_selected_points_1) > 0:
                # input_points_31 = np.array(all_selected_points_1)
                input_points_31 =  np.array(all_selected_points_1).reshape(-1, 1, 2)
                # print("Points after selection:", input_points_31)

                # Generate new segmentation map and compute IoU
                seg_map_ = testing_loop(input_points_31)
                iou = calculate_iou(seg_map_, gt_dilated_mask)
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

    # if best_iou<lower_bound:
    #     rep_points= process_single_image_using_rep_point_logic(gt_dilated_mask)
    #     print("Rep points",rep_points)
    #     # input_points_41 = np.array(rep_points)
    #     rep_points_arr = []

    #     for coord in rep_points:
    #         transformed = [[int(coord[0][0]), int(coord[0][1])]]
    #         # print("transformed rep points",transformed)
    #         rep_points_arr.append(transformed)

    #     # Convert the list to a numpy array
    #     rep_array = np.array(rep_points_arr)
    #     print("Points after rep point logic:", rep_array)
    #     seg_map_ = testing_loop(rep_array)
    #     print("ground truth mask shape",gt_dilated_mask.shape)
    #     iou = calculate_iou(seg_map_, gt_dilated_mask)
    #     print("Updated IoU after representative point :", iou)
    #     if iou > best_iou:
    #         best_iou = iou
    #         best_seg_map = seg_map_
    #         best_prompt = rep_array  # Update best prompt
    #         print("Best IoU updated:", best_iou)


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
        rep_array = np.array(transformed_group, dtype=np.int32)
        print("Points after rep point logic and foreground check:", rep_array)
    
        # Process the transformed points
        seg_map_ = testing_loop(rep_array)
        print("Ground truth mask shape:", gt_dilated_mask.shape)
        iou = calculate_iou(seg_map_, gt_dilated_mask)
        print("Updated IoU after representative point:", iou)
        
        if iou > best_iou:
            best_iou = iou
            best_seg_map = seg_map_
            best_prompt = rep_array  # Update best prompt
            print("Best IoU updated:", best_iou)


    txt_filename = os.path.join(output_txt_files_dir, f"{os.path.splitext(img_name)[0]}.txt")

    
    # # Save the best prompt and IoU to the text file
    with open(txt_filename, "w") as file:
       
        for point in best_prompt:
            for pt in point:
                # if len(point) == 2:
                file.write(f"{pt[0]},{pt[1]}\n")
             
    
    cv2.imwrite(f"{output_masks_dir}/{img_name}",best_seg_map)
   
   
    
















































