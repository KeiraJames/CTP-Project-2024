"""
    IGNORE FOR NOW
"""
import numpy as np
import cv2
from scipy.signal import convolve2d
import os

def wiener_filter(img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    img_filtered = np.zeros_like(img)
    
    # Apply Wiener filter for each color channel
    for i in range(img.shape[2]):
        img_filtered[:, :, i] = convolve2d(img[:, :, i], kernel, 'same')
    
    return img_filtered

def apply_wiener_filter_to_images(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            print(f"Processing: {filename}")
            
            # Read the color image
            img = cv2.imread(image_path)

            if img is None:
                print(f"Error loading {filename}")
                continue

            # Apply Wiener filter
            img_deblurred = wiener_filter(img)

            # Clip values to valid range [0, 255]
            img_deblurred = np.clip(img_deblurred, 0, 255).astype(np.uint8)

            # Save the deblurred image
            output_path = os.path.join(folder, f'deblurred_{filename}')
            cv2.imwrite(output_path, img_deblurred)
            print(f"Saved deblurred image: {output_path}")

if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    apply_wiener_filter_to_images(input_folder)
