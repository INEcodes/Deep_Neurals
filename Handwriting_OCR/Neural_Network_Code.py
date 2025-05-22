import os

folder = 'Handwriting_OC/Dataset/data_subset'
image_files = os.listdir(folder)
print(f"Total images found: {len(image_files)}")
print(image_files[:10])  # print first 10 filenames