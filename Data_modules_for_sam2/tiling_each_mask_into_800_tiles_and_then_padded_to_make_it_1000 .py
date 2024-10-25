import os
import cv2

# Define your padding and tiling functions
def pad_image_to_tile_size(image, tile_size):
    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) if h % tile_size != 0 else 0
    pad_w = (tile_size - w % tile_size) if w % tile_size != 0 else 0
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def split_image_into_tiles(image, tile_size, pad_size):
    h, w = image.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            top_pad = (pad_size - tile.shape[0]) // 2
            bottom_pad = pad_size - tile.shape[0] - top_pad
            left_pad = (pad_size - tile.shape[1]) // 2
            right_pad = pad_size - tile.shape[1] - left_pad
            padded_tile = cv2.copyMakeBorder(tile, top_pad, bottom_pad, left_pad, right_pad, 
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
            tiles.append(padded_tile)
    return tiles

def process_image(image):
    tile_size = 800
    pad_size = 1000
    padded_image = pad_image_to_tile_size(image, tile_size)
    tiles = split_image_into_tiles(padded_image, tile_size, pad_size)
    return tiles

# Main processing function
def process_mask_directory(root_mask_dir, output_dir):
    for subdir in os.listdir(root_mask_dir):  # For each subdirectory in the root mask directory
        subdir_path = os.path.join(root_mask_dir, subdir)
        
        if not os.path.isdir(subdir_path):
            continue  # Skip files, we only want directories
        
        # Now process each mask inside the subdirectory
        for mask_filename in os.listdir(subdir_path):
            if mask_filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if necessary
                mask_path = os.path.join(subdir_path, mask_filename)
                image = cv2.imread(mask_path)

                if image is None:
                    print(f"Error loading mask: {mask_path}")
                    continue

                # Process the mask image to get tiles
                tiles = process_image(image)

                # Create the output directory for this particular mask
                mask_output_dir = os.path.join(output_dir, subdir, os.path.splitext(mask_filename)[0])
                os.makedirs(mask_output_dir, exist_ok=True)

                # Save each tile
                for idx, tile in enumerate(tiles):
                    tile_output_path = os.path.join(mask_output_dir, f"{os.path.splitext(mask_filename)[0]}_tile_{idx}.jpg")
                    cv2.imwrite(tile_output_path, tile)
                    print(f"Tile {idx} saved at: {tile_output_path}")

# Example usage
root_mask_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/masks/'  # Root directory containing subdirectories of masks
output_dir = '/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/tiling_masks_outputs/'  # Output directory for saving the tiles
process_mask_directory(root_mask_dir, output_dir)
