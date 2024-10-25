import cv2
import numpy as np
import os

def pad_image_to_tile_size(image, tile_size):
    """
    Pads the image so its dimensions are divisible by the tile size.
    """
    h, w = image.shape[:2]
    pad_h = (tile_size - h % tile_size) if h % tile_size != 0 else 0
    # print("pad_h:",pad_h)
    pad_w = (tile_size - w % tile_size) if w % tile_size != 0 else 0
    # print("pad_w:",pad_w)
    # Apply padding to the bottom and right of the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image




def split_image_into_tiles(image, tile_size, pad_size):
    """
    Splits the padded image into 800x800 tiles, then pads each tile symmetrically to 1000x1000.
    """
    h, w = image.shape[:2]
    tiles = []
    
    # Iterate through the image and extract 800x800 tiles
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image[y:y+tile_size, x:x+tile_size]
            
            # Calculate the padding for each side to pad to 1000x1000
            top_pad = (pad_size - tile.shape[0]) // 2
            # print("top pad:",top_pad)
            bottom_pad = pad_size - tile.shape[0] - top_pad
            # print("bottom pad:",top_pad)
            left_pad = (pad_size - tile.shape[1]) // 2
            # print("left pad:",top_pad)
            right_pad = pad_size - tile.shape[1] - left_pad
            # print("right pad:",top_pad)
            
            # Apply padding symmetrically
            padded_tile = cv2.copyMakeBorder(tile, top_pad, bottom_pad, left_pad, right_pad, 
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
            tiles.append(padded_tile)
    
    return tiles



def process_image(image):
    # Load the high-resolution image
    # image = cv2.imread(image_path)

    # Define the tile size (800x800) and pad size (1000x1000)
    tile_size = 800
    pad_size = 1000

    # Step 1: Pad the image to make its dimensions divisible by 800
    padded_image = pad_image_to_tile_size(image, tile_size)

    # Step 2: Split the image into tiles and pad each tile to 1000x1000
    tiles = split_image_into_tiles(padded_image, tile_size, pad_size)

    return tiles

directory_path = "/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/data/"
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if necessary
        image_path = os.path.join(directory_path, filename)
        
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Process the image to get the tiles
        tiles = process_image(image)
        
        for idx, tile in enumerate(tiles):
            # Creating a directory for the output tiles of each image
            image_output_dir = os.path.join('/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/', os.path.splitext(filename)[0])
            image_output_dir = f"{image_output_dir}_tiles"
            os.makedirs(image_output_dir, exist_ok=True)

            # Get filename without extension
            base_filename = os.path.splitext(filename)[0]
            
            # Constructing the full path for each tile
            image_output_dir_path = os.path.join(image_output_dir, f"{base_filename}_{idx}.jpg")
            
            # Saving the tile
            cv2.imwrite(image_output_dir_path, tile)
            print(f"Tile {idx} saved at: {image_output_dir_path}")
            # print(f"Tile {idx} saved at: {image_output_dir_path}")

















# def split_image_into_tiles(image, tile_size, pad_size):
#     """
#     Splits the padded image into 800x800 tiles, then pads each tile to 1000x1000.
#     """
#     h, w = image.shape[:2]
#     tiles = []
    
#     # Iterate through the image and extract 800x800 tiles
#     for y in range(0, h, tile_size):
#         for x in range(0, w, tile_size):
#             tile = image[y:y+tile_size, x:x+tile_size]
            
#             # Apply padding to make it 1000x1000
#             padded_tile = cv2.copyMakeBorder(tile, 0, pad_size - tile.shape[0], 0, pad_size - tile.shape[1], 
#                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
#             tiles.append(padded_tile)
    
#     return tiles





# for filename in os.listdir(directory_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Add more extensions if necessary
#             image_path = os.path.join(directory_path, filename)
#             print(f"Processing image: {image_path}")
#             image = cv2.imread(image_path)
#             tiles = process_image(image)
#             for idx, tile in enumerate(tiles):
#                 image_output_dir = os.path.join('/media/usama/SSD/Usama_dev_ssd/clewiston_masks_24/', os.path.splitext(filename)[0])
#                 os.makedirs(image_output_dir, exist_ok=True)
#                 image_output_dir_path = os.path.join(image_output_dir,f"Mask_tiles_clewiston/tile_{idx}.jpg")
#                 cv2.imwrite(image_output_dir_path, tile)





            # if image is not None:
                # Create a unique output directory for each image
                

# image_path = '/home/usama/masks/127.jpg'  # Replace with your image path
# tiles = process_image(image_path)

# To save tiles (optional)
# for idx, tile in enumerate(tiles):
#     cv2.imwrite(f'/media/usama/SSD/Usama_dev_ssd/24_oct_code_sam2/Mask_tiles_clewiston/tile_{idx}.jpg', tile)
