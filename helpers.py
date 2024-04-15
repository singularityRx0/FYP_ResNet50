import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from random import shuffle
import cv2
import os
import shutil
import random

import __transform_split as ts

def visualize_total_image(DATA_PATH):
    class_names = os.listdir(DATA_PATH)
    image_count = {}
    total_image = 0

    for i in class_names:
        images_in_class= len(os.listdir(os.path.join(DATA_PATH,i)))
        image_count[i] = images_in_class
        total_image += images_in_class

    ax1 = plt.subplot()
    ax1.pie(image_count.values(), labels = image_count.keys(), shadow = False, autopct = '%1.1f%%', startangle=90)
    ax1.set_title('Total Images: {}'.format(total_image))
    plt.show()


def visualize_class_images(DATA_PATH):
    check_list = [('1','COVID19 Negative-Healthy'), ('2','COVID19 Positive'), ('3','Community Aquired Pneumonia (CAP)')]
    fig = plt.figure(figsize=(16,5))

    if check_list[0][0] in DATA_PATH:
        fig.suptitle(check_list[0][1], size=22)
    elif check_list[1][0] in DATA_PATH:
        fig.suptitle(check_list[1][1], size=22)
    elif check_list[2][0] in DATA_PATH:
        fig.suptitle(check_list[2][1], size=22)

    img_paths = os.listdir(DATA_PATH)
    shuffle(img_paths)

    for i, image in enumerate(img_paths[:4]):
        img = cv2.imread(os.path.join(DATA_PATH, image))
        plt.subplot(1, 4, i+1, frameon=False)
        plt.imshow(img)
    plt.show()

def visualize_split_transform_images(input, size=(16,10), title=None):
    input = input.numpy().transpose((1,2,0))
    mean = ts.mean_nums
    std = ts.std_nums
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.figure(figsize=size)
    plt.imshow(input)
    if title is not None:
        plt.title(title, size=22)
    plt.pause(0.001)
    plt.show()

def copy_images(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        dest_path = os.path.join(destination_folder, os.path.relpath(root, source_folder))
        os.makedirs(dest_path, exist_ok=True)

        # Count the number of images copied from each folder
        images_copied = 0
        
        for file in files:
            print("Copying file {}.".format(file))
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Copy the file to the destination folder
                shutil.copy2(os.path.join(root, file), dest_path)
                images_copied += 1
                # Stop copying after 10 images are copied from this folder
                if images_copied >= 10:
                    break

def move_images(source_folder, destination_folder):  
    # Iterate through subfolders in the source folder
    for root, dirs, files in os.walk(source_folder):
        # Iterate through files in each subfolder
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                print("Moving image {}.".format(file))
                # Construct source and destination paths
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                # Move the file to the destination folder
                shutil.move(source_path, destination_path)
    # shutil.rmtree(root)

def move_random_images(source_folder, destination_folder, num_images=4403):
    # List all files in the source folder
    files = os.listdir(source_folder)
    
    # Filter out non-image files (you can adjust this if needed)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Randomly select num_images files
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    # Move selected files to the destination folder
    for file in selected_files:
        print("moving {} to overfilled".format(file))
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)

if __name__ == "__main__":
    # destination_folder = "C:/Users/PC/Documents/0.Personal/FYP/FYP/Code/Train"
    # source_folder = "C:/Users/PC/Pictures/COVID-19 Database/Synthetic_COVID19_dataset"

    # source_folder = "C:/Users/PC/Documents/0.Personal/FYP/FYP/Code/Train/Synthetic"
    # destination_folder = "C:/Users/PC/Documents/0.Personal/FYP/FYP/Code/Train/2COVID"

    destination_folder = "C:/Users/PC/Documents/0.Personal/FYP/FYP/Code/Train/coivd-overfilled"
    source_folder = "C:/Users/PC/Documents/0.Personal/FYP/FYP/Code/Train/2COVID"

    move_random_images(source_folder,destination_folder)

# def plot(imgs, row_title=None, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             boxes = None
#             masks = None
#             if isinstance(img, tuple):
#                 img, target = img
#                 if isinstance(target, dict):
#                     boxes = target.get("boxes")
#                     masks = target.get("masks")
#                 elif isinstance(target, tv_tensors.BoundingBoxes):
#                     boxes = target
#                 else:
#                     raise ValueError(f"Unexpected target type: {type(target)}")
#             img = F.to_image(img)
#             if img.dtype.is_floating_point and img.min() < 0:
#                 # Poor man's re-normalization for the colors to be OK-ish. This
#                 # is useful for images coming out of Normalize()
#                 img -= img.min()
#                 img /= img.max()

#             img = F.to_dtype(img, torch.uint8, scale=True)
#             if boxes is not None:
#                 img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
#             if masks is not None:
#                 img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

#             ax = axs[row_idx, col_idx]
#             ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])

#     plt.tight_layout()