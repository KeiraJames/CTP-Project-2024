#################################################################
# Filename: convert_images.py
# Decription: convert images to jpg format
#################################################################

# Import libraries 

#operating system module for operating system dependent functionality
import os

#python image library for image processing
from PIL import Image
###################################################################

def convert_images(folder):
    # Loop through the image folder directory
    for filename in os.listdir(folder):
        # Check if the file is not in JPG format
        if not filename.lower().endswith('.jpg') and filename.lower().endswith(('.png', '.gif', '.bmp', '.jpeg')):
            input_path = os.path.join(folder, filename)
            output_path = os.path.join(folder, f"{os.path.splitext(filename)[0]}.jpg") #jpg converted path

            try:
                # Open the image file
                with Image.open(input_path) as img:
                    # Convert the image to RGB
                    rgb_img = img.convert('RGB')
                    # Save image as JPG
                    rgb_img.save(output_path, 'JPEG')
                    print(f"Converted {filename} to {output_path}")
                    # Remove the old image file
                    os.remove(input_path)
                    print(f"Removed old file: {input_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print("Image conversion to .jpg completed.")  # Print once after processing all images

#####################################################################################################################

if __name__ == '__main__':
    input_folder = input("Enter the path to the input folder containing images: ")
    convert_images(input_folder)
