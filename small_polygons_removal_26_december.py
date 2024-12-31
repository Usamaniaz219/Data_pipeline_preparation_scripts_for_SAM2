
import cv2
import numpy as np
import os

def remove_small_polygons(binary_image, area_threshold):

    # Find contours of connected components
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw filtered contours
    filtered_image = np.zeros_like(binary_image)

    for contour in contours:
        # Calculate area of the contour
        area = cv2.contourArea(contour)

        # Retain only polygons with an area above the threshold
        if area >= area_threshold:
            cv2.drawContours(filtered_image, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_image

def load_binary_image(image_path):
    # Load the image in grayscale
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the image is binary (0 and 255 values only)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    return binary_image


# Function to load a binary image from a specified directory

if __name__ == "__main__":
    # Define the directory containing images
    mask_directory = "/home/usama/wajeeha_dataset_2/sam2_zone_based_dataset_Final_small_polygons_to_remove/"

    mask_files = [f for f in os.listdir(mask_directory) if f.endswith('.jpg') or f.endswith('.png')]
    for mask_file in mask_files:
        print(mask_file)
        mask_path = os.path.join(mask_directory, mask_file)

        # Load the binary image
        binary_image = load_binary_image(mask_path)

        # Define area threshold
        area_threshold =200

        # Remove small polygons
        cleaned_image = remove_small_polygons(binary_image, area_threshold)
        cv2.imwrite(f"/home/usama/wajeeha_dataset_2/removed_small_polygons_sam2_zone_based_dataset/{mask_file}",cleaned_image)
        

        cv2.imshow("Cleaned Image", cleaned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()