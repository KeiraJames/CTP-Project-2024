import pandas

import os

from PIL import Image

#none of the images are corrupted
def is_corrupt(image_path):
    try:
        img = Image.open(image_path)
        img.verify()  # Verify the image file
        return False  # Image is not corrupted
    except (IOError, SyntaxError) as e:
        return True  # Image is corrupted

def read_files_in_folder(folder_path):
    count=0
    for filename in os.listdir(folder_path):
         file_path = os.path.join(folder_path, filename)
         if is_corrupt(file_path):
            count+=1;
            # print("Image is corrupted:", file_path)
        # else:
        #       print("Image is OK:", file_path)
    return count;
print("train/FAKE")
folder_path = "archive/train/FAKE"  # Replace with your folder path
print(read_files_in_folder(folder_path)) 
print("train/Real")
print(read_files_in_folder("archive/train/REAL"))
print("test/fake")
print(read_files_in_folder("archive/test/FAKE"))
print("test/real")
print(read_files_in_folder("archive/test/REAL"))
