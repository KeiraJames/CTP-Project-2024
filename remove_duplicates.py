import os
import hashlib
from PIL import Image

def calculate_hash(image_path):

    #Calculate the hash of an image.
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB format
        img = img.resize((8, 8))  # Resize to reduce size and create hash
        hash_value = hashlib.md5(img.tobytes()).hexdigest()  # Create hash
    return hash_value

def find_and_remove_duplicates(folder_path):

    #Find and remove duplicate images in a given folder.

    #If cannot find path/ folder, Print that it does not exist
    if not os.path.exists(folder_path):

        print(f"The folder '{folder_path}' may not exist.")
        r
        eturn

    print(f"Scanning folder: {folder_path}")

    hashes = {}
    duplicates = []

    for filename in os.listdir(folder_path):# for each file in the folder

        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):# if file is an image

            file_path = os.path.join(folder_path, filename) #generate a path to the specific image

            print(f"Processing file: {file_path}")  # Debug output

            img_hash = calculate_hash(file_path)

            if img_hash in hashes:
                duplicates.append(file_path)  # Found a duplicate
                print(f"Duplicate found: {file_path} (duplicate of {hashes[img_hash]})")
            else:
                hashes[img_hash] = file_path

    # Remove duplicates
    for duplicate in duplicates:

        os.remove(duplicate)
        print(f"Removed duplicate: {duplicate}")

    if not duplicates:
        print("No duplicates found.")

if __name__ == '__main__':
    folder = input("Enter the path to the folder containing photos: ")
    find_and_remove_duplicates(folder)

