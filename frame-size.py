import os

def count_images(train_folder_path):
    image_count = 0
    for root, directories, files in os.walk(train_folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                image_count += 1

    return image_count

train_folder_path = "UCF101_subset/val"
image_count = count_images(train_folder_path)
print("Total images:", image_count)
