import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchinfo import summary

import matplotlib.pyplot as plt
import time
import copy
from random import shuffle

from tqdm.notebook import tqdm

import sklearn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import classification_report
from PIL import Image
import cv2

import os
import shutil

#Statistics Based on ImageNet Data for Normalisation
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

data_transforms = {"train":transforms.Compose([
                                transforms.Resize((224,224)), #Resizes all images into same dimension
                                transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
                                transforms.RandomRotation(10), # Rotates the images upto Max of 10 Degrees
                                transforms.RandomHorizontalFlip(p=0.4), #Performs Horizantal Flip over images 
                                transforms.ToTensor(), # Coverts into Tensors
                                transforms.Normalize(mean = mean_nums, std=std_nums)]), # Normalizes
                    "val": transforms.Compose([
                                transforms.Resize((224,224)),
                                transforms.CenterCrop(224), #Performs Crop at Center and resizes it to 150x150
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_nums, std = std_nums)
                    ])}

def load_split_train_test(DATA_PATH, valid_size = .2):
    since = time.time()

    train_data = datasets.ImageFolder(DATA_PATH, transform=data_transforms['train']) #Picks up Image Paths from its respective folders and label them
    test_data = datasets.ImageFolder(DATA_PATH, transform=data_transforms['val'])
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    
    dataset_size = {"train":len(train_idx), "val":len(test_idx)}
    train_sampler = SubsetRandomSampler(train_idx) # Sampler for splitting train and val images
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=8) # DataLoader provides data from traininng and validation in batches
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=8)
    
    time_since = time.time() - since

    if not trainloader or not testloader:
        load_split_train_coor = 0
    else:
        load_split_train_coor = 1
        print('-' * 10)
        print('Successfully transformed and splitted dataset')
        print('Data load completed in {:.0f}m {:.0f}s'.format(time_since // 60, time_since % 60))
        print('-' * 10)
        print()

    return trainloader, testloader, dataset_size
    
def load_siamese_pair(DATA_PATH, valid_size = .2):
    train_data = datasets.ImageFolder(DATA_PATH, transform=data_transforms['train'])
    test_data = datasets.ImageFolder(DATA_PATH,transform=data_transforms['val'])

    
