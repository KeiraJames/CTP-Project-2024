###########################################################################
# Filename: enhance_image.py
# Description: enhances image contrast by applying gemma correction
##########################################################################

# import libraries

# computer vision and machine learning library
import cv2

# library for numerical operation
import numpy as np

# operating system module for operating system dependent functionality
import os

#############################################################################################################

# Gemma correction function to fix the contrast by fixing overall distribution of pixel intensities in the image.
def gamma_correction(image, gamma=1.5):
    # Convert to grayscale if it's a colored image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply gamma correction
    adjusted_image = np.power(gray_image / 255.0, gamma) * 255
    
    return adjusted_image.astype(np.uint8)

def contrast_correction(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Apply histogram equalization (or any other enhancement technique)
            enhanced_image = gamma_correction(image)

            # Save the enhanced image
            enhanced_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(enhanced_image_path, enhanced_image)

################################################################################################################

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder containing images: ")
    output_folder = input("Enter the path to the output folder: ")
    contrast_correction(input_folder, output_folder)
