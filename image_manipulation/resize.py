############################################################################################
# Filename: resize.py
# Description: Resizes image to certain dimention
############################################################################################

# import libraries

#python image library for image processing
from PIL import Image

#operating system module for operating system dependent functionality
import os

# library for mathematical computation
import numpy as np

# computer vision and machine learning library
import cv2

############################################################################################

def resize(folder, size=(128, 128)):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            print(f"Processing: {filename}") 
            # load the image
            img = cv2.imread(image_path)
            # resizes the image 
            img_resized = cv2.resize(img, size)
            # saves the image
            cv2.imwrite(image_path, img_resized)

############################################################################################
                
if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder containing images: ")
    resize(input_folder)
