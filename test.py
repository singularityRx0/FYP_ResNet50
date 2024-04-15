from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import v2
import torchvision

import __data_loader as dl
import __transform_split as ts
import __gpu as gpu
import helpers

folder_name = "Train"

DATA_PATH, NORMAL_PATH, COVID_PATH, CAP_PATH = dl.load_data(folder_name)
helpers.visualize_total_image(DATA_PATH)
# # helpers.visualize_class_images(CAP_PATH)


trainloader, valloader, dataset_size = ts.load_split_train_test(DATA_PATH, .3)
dataloaders = {"train":trainloader, "val":valloader}
data_sizes = {x: len(dataloaders[x].sampler) for x in ['train','val']}
class_names = trainloader.dataset.classes
print(data_sizes)

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
helpers.visualize_split_transform_images(out, title=[class_names[x] for x in classes])


# torch.cuda.empty_cache() 











# plt.rcParams["savefig.bbox"] = 'tight'

# # if you change the seed, make sure that the randomly-applied transforms
# # properly show that the image can be both transformed and *not* transformed!
# torch.manual_seed(0)

# total_img = []

# orig_img = Image.open(Path('C:/Users/PC/Documents/FYP/FYP/Code/SARS-COV-2 Ct-Scan Dataset/1NonCOVID') / 'Non-Covid (1).png')
# gray_img = v2.Grayscale(num_output_channels=3)(orig_img)

# jitter = v2.ColorJitter(brightness=.5, hue=.3)(orig_img)


# total_img.append(orig_img)
# total_img.append(jitter)

# # Create subplots
# num_images = len(total_img)
# fig, axes = plt.subplots(1, num_images, figsize=(10, 5))

# # Plot each image
# for i in range(num_images):
#     axes[i].imshow(total_img[i])
#     axes[i].axis('off')  # Hide axis
#     if i ==0:
#         axes[i].set_title(f'Original Image {i+1}')
#     else:
#         axes[i].set_title(f'Image {i+1}')

# # Adjust layout
# plt.tight_layout()


# plt.show()