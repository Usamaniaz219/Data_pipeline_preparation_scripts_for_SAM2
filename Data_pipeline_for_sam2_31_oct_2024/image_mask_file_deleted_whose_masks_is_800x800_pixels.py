import os
import cv2

def delete_files_if_mask_matches(mask_dir, image_dir, txt_dir):
    """
    Deletes mask files, corresponding image files, and text files if the mask's foreground area matches the target size.
    
    :param mask_dir: Directory containing mask files.
    :param image_dir: Directory containing image files.
    :param txt_dir: Directory containing text files.
    :param target_size: Tuple indicating the width and height of the target foreground area.
    """
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    for mask_file in mask_files:
        # Construct the file paths
        mask_path = os.path.join(mask_dir, mask_file)
        base_name = os.path.splitext(mask_file)[0]
        image_path = os.path.join(image_dir, base_name + '.jpg')
        txt_path = os.path.join(txt_dir, base_name + '.txt')

        # Check if corresponding image and text file exist
        if not os.path.exists(image_path) or not os.path.exists(txt_path):
            continue

        # Read the mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Unable to read mask file: {mask_path}")
            continue

        # Calculate the foreground area
        foreground_area = cv2.countNonZero(mask)
        if foreground_area >= 640000:
            # Delete the files if the condition is met
            os.remove(mask_path)
            os.remove(image_path)
            os.remove(txt_path)
            print(f"Deleted: {mask_path}, {image_path}, {txt_path}")

mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_1/masks/'
image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_1/images/'
txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/training_data_sam2_22/train_data_1/txt_files/'

delete_files_if_mask_matches(mask_dir, image_dir, txt_dir)
