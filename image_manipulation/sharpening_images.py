###########################################################################
# Filename: shapening_images.py
# Description: sharpens images by applying kernel to the images
##########################################################################

# import libraries

# computer vision and machine learning library
import cv2

# library for numerical operation
import numpy as np

# operating system module for operating system dependent functionality
import os

##########################################################################

def sharpen_images(folder):
    # sharpening kernel
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            print(f"Processing: {filename}") 
            
            #load image
            image = cv2.imread(image_path)

            if image is not None:
                #Apply sharpening kernel
                sharpened_image = cv2.filter2D(image, -1, kernel)

                # Display the original and sharpened images
                # cv2.imshow('Original Image', image)
                # cv2.imshow('Sharpened Image', sharpened_image)

                # Wait for a key press to close the windows
                # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                # cv2.destroyAllWindows()  # Close all OpenCV windows

                # save the sharpened image by overwriting the original blurry image
                cv2.imwrite(image_path, sharpened_image) 

                print(f"Saved sharpened image: {image_path}")
            else:
                # Print error message if image could not be loaded
                print(f"Failed to load image: {image_path}")

    print("Processing complete")
    
##########################################################################
if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder containing images: ")
    sharpen_images(input_folder)

