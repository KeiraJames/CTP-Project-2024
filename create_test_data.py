import os
import shutil
import random

source_directory = input("Enter source directory: ")
destination_directory = input("Enter destinaton directory: ")

#get the total number of files in the directory
count = 0
for file in os.listdir(source_directory):
    all_files = file
    count += 1

#get the list of files
all_files = os.listdir(source_directory)

#get percentage of files to move and sample
twenty_percent = count//5

files_to_move = random.sample(all_files, twenty_percent)


for each_file in files_to_move:
    source_file = os.path.join(source_directory, each_file)
    destination_file = os.path.join(destination_directory, each_file)
    
    # move the file
    shutil.move(source_file, destination_file)       
