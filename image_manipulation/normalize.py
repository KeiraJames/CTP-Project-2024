############################################################################################
# Filename: normalize.py
# Description: Normalizes the image (makes pixel value consistent accross image)
############################################################################################

# import libraries

#operating system module for operating system dependent functionality
import os

# library for mathematical computation
import numpy as np

# computer vision and machine learning library
import cv2
############################################################################################
def normalize(folder, size=(128, 128)):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            print(f"Processing: {filename}")
            # load the image
            img = cv2.imread(image_path) 
            #normalize the image
            img_normalized = img / 255.0 
            # scales it back to 0 to 255 range
            img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)
            # save the image
            cv2.imwrite(image_path, img_normalized_uint8)
############################################################################################
                
if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder containing images: ")
    normalize(input_folder)
