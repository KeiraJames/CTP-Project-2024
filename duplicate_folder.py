import os
import hashlib
from PIL import Image

def calculate_hash(image_path):
    # Calculate the hash of an image.
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB format
        img = img.resize((8, 8))  # Resize to reduce size and create hash
        hash_value = hashlib.md5(img.tobytes()).hexdigest()  # Create hash
    return hash_value

def find_and_compare_duplicates(folder1, folder2):
    # Find and remove duplicate images between two folders.

    # Check if both folders exist
    if not os.path.exists(folder1):
        print(f"The folder '{folder1}' does not exist.")
        return
    if not os.path.exists(folder2):
        print(f"The folder '{folder2}' does not exist.")
        return

    print(f"Scanning folders: {folder1} and {folder2}")

    hashes = {}
    duplicates = []

    # Process images in the first folder
    for filename in os.listdir(folder1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder1, filename)
            print(f"Processing file in folder 1: {file_path}")
            img_hash = calculate_hash(file_path)
            hashes[img_hash] = file_path

    # Process images in the second folder and check for duplicates
    for filename in os.listdir(folder2):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(folder2, filename)
            print(f"Processing file in folder 2: {file_path}")
            img_hash = calculate_hash(file_path)

            if img_hash in hashes:
                duplicates.append((file_path, hashes[img_hash]))
                print(f"Duplicate found: {file_path} (duplicate of {hashes[img_hash]})")
                # Remove duplicate from folder 2
                os.remove(file_path)
                print(f"Removed duplicate: {file_path}")

    if not duplicates:
        print("No duplicates found between the two folders.")
    else:
        print("\nList of duplicates that were removed:")
        for dup, original in duplicates:
            print(f"Duplicate: {dup} (duplicate of {original})")

if __name__ == '__main__':
    folder1 = input("Enter the path to the first folder: ")
    folder2 = input("Enter the path to the second folder: ")
    find_and_compare_duplicates(folder1, folder2)
