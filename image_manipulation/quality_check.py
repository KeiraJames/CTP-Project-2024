########################################################################################
# Filename: quality_check.py
# Description: checks the quality of the image and returns issue
########################################################################################

# import library

# computer vision and machine learning library
import cv2

# library for numerical operation
import numpy as np

# library for mathematical purpose
import matplotlib.pyplot as plt

# operating system module for operating system dependent functionality
import os

########################################################################################

# assess the contrast quality of each image (overall distribution of pixel intensities in the image.)
def check_histogram_quality(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_sum = hist.sum()
    hist_normalized = hist / hist_sum
    hist_std = hist_normalized.std()
    return hist_std

# checks the sharpness level of each image by applying Laplacian algorithm
def check_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# checks the mean variance of each image
def check_mean_variance(gray):
    mean_intensity = np.mean(gray)
    variance_intensity = np.var(gray)
    return mean_intensity, variance_intensity

# Returns result based on the quality of each image
def check_image_quality(folder):
    results = []  # Collect results for all images
    for filename in os.listdir(folder):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder, filename)
            print(f"Processing: {filename}") 
            image = cv2.imread(image_path)
            if image is None:
                results.append(f"{filename}: Error: Image not found.")
                continue  # Skip to the next image

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Quality assessments
            hist_std = check_histogram_quality(gray)
            sharpness = check_sharpness(gray)
            mean_intensity, variance_intensity = check_mean_variance(gray)

            quality_issues = []

            print(f"hist_std for {image_path}: {hist_std}")

            #Histogram quality check
            if hist_std <= 0.1:
                quality_issues.append("Histogram variance is low; consider improving contrast.")
            
            # Sharpness check
            if sharpness < 100:  # Adjust as necessary
                quality_issues.append("Image is blurry; consider sharpening.")

            # Mean intensity check
            if mean_intensity <= 50:
                quality_issues.append("Image may be underexposed; consider brightening.")
            elif mean_intensity >= 200:
                quality_issues.append("Image may be overexposed; consider reducing brightness.")
            
            # Variance check
            if variance_intensity < 1000:  # Adjust threshold as necessary
                quality_issues.append("Image has low intensity variance; check for flat areas.")

            # Report results for this image
            if quality_issues:
                results.append(f"{filename}: Image quality is not satisfactory. Issues found:\n- " + "\n- ".join(quality_issues))
            else:
                results.append(f"{filename}: Image quality is good.")

    return "\n".join(results)  # Return results for all images


if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder containing images: ")
    result = check_image_quality(input_folder)
    print(result)
