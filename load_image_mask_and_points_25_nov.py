# import cv2
# import numpy as np
# import os

# def load_image_mask_and_points(image_dir, mask_dir, txt_dir):
#     # List all images in the image directory
#     image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
#     # Select a random image
#     random_index = np.random.randint(len(image_files))
#     image_filename = image_files[random_index]
#     for image_filename in image_files:
#         print("image filename",image_filename)
        
#         # Construct corresponding mask and txt file paths
#         mask_filename = os.path.join(mask_dir, image_filename)  # Assuming mask has the same name as the image
#         print("mask filename",mask_filename)
#         txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
#         # print("txt_file_name",txt_filename)
#         # Read image and mask
#         Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Read image
#         ann_map = cv2.imread(mask_filename)  # Read annotation (mask)
        
#         # Read points from the text file
#         with open(txt_filename, 'r') as f:
#             points = [list(map(int, line.strip().split(','))) for line in f.readlines()]
#             print("points",type(points))

#         # Resize image and mask
#         r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
#         Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
#         ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)))


#         return Img, np.array(ann_map), np.array(points), np.ones([len(points), 1])

# # Example usage
# image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/images/'
# mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/masks/'
# txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/txt_files/'

# # Call the function
# Img, mask, points, labels = load_image_mask_and_points(image_dir, mask_dir, txt_dir)
# # cv2.imwrite("mask.jpg",mask)
# # cv2.imwrite("image.jpg",Img)
# # print("points",points)
# # print("label:",labels)
# # print("type of mask",type(mask))
# # print("type of image",type(Img))




import os
import numpy as np
import cv2

def load_image_mask_and_points(image_dir, mask_dir, txt_dir):
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Iterate over all image files
    for image_filename in image_files:
        print("Processing image:", image_filename)
        
        # Construct corresponding mask and txt file paths
        mask_filename = os.path.join(mask_dir, image_filename)  # Assuming mask has the same name as the image
        print("Mask filename:", mask_filename)
        txt_filename = os.path.join(txt_dir, os.path.splitext(image_filename)[0] + '.txt')  # Change extension to .txt
        
        # Read image and mask
        Img = cv2.imread(os.path.join(image_dir, image_filename))[..., ::-1]  # Read image
        ann_map = cv2.imread(mask_filename)  # Read annotation (mask)
        
        # Read points from the text file
        with open(txt_filename, 'r') as f:
            points = [list(map(int, line.strip().split(','))) for line in f.readlines()]
            print("Points:", points)

        # Resize image and mask
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # Scaling factor
        Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
        ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)))

        # Yield the processed image, mask, points, and labels
        yield Img, np.array(ann_map), np.array(points), np.ones([len(points), 1])

# Example usage
image_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/images/'
mask_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/masks/'
txt_dir = '/media/usama/SSD/Data_for_SAM2_model_Finetuning/notebook/output/txt_files/'

# Create a generator
data_generator = load_image_mask_and_points(image_dir, mask_dir, txt_dir)

# Iterate through the generator to process each image one at a time
for Img, mask, points, labels in data_generator:
    cv2.imwrite("image.jpg",Img)
    cv2.imwrite("mask.jpg",mask)
    print("Image shape:", Img.shape)
    print("Mask shape:", mask.shape)
    print("Points:", points)
    print("Labels:", labels)
    # You can perform further processing here or save the images/masks as needed