import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm

from wikiart import WikiArtDataset, CNNWikiArt
from TrainerWiki import Trainer

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight


trainingdir = '/scratch/lt2326-2926-h24/wikiart/train'


device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
deivce = torch.cuda.current_device()


print("Running...")


dataset = WikiArtDataset(trainingdir, device)


### get minority labels
labels = [label for _, label in dataset]
tot_number_of_samples = len(labels)
# Count occurrences of each class
class_counts = Counter(labels)

minority_labels = []
for key in class_counts.keys(): 
    if (class_counts[key] / tot_number_of_samples) < 0.01: 
        minority_labels.append(key)

print(minority_labels)


### Measure 1: 
# Define transformations for data augmentation
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),])

# Normalize after augmentation
augmentation = transforms.Compose([
    augmentation,
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

# Need to reload the dataset to apply augmentation
dataset = WikiArtDataset(trainingdir, device, augmentation , minority_labels)



### Measure 2: Use class weights; they will be loaded with the model.
# Compute class weights using inverse frequency
num_of_labels = len(class_counts)
class_weights = compute_class_weight('balanced', classes=np.arange(num_of_labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(dataset, labels):
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)


### Measure 3: Oversample the minority classes
# Extract labels from training data for sampler
sample_weights = [1 / class_counts[label] for _, label in train_data]

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),  # Number of samples to draw in an epoch
    replacement=True          # Allow resampling
)


cnn_class_imbalance = CNNWikiArt(300, output_size=num_of_labels, optimizer = 'Adam', learning_rate = 0.001, l2 = 0.0, loss_function = 'CEL', class_weights = class_weights).to(device)
print(next(cnn_class_imbalance.parameters()).device)
trainer = Trainer(max_epochs = 15, batch_size = 64, early_stopping_patience = 2, min_delta = 0.005, sampler = sampler )
trainer.fit(cnn_class_imbalance,train_data,val_data)

torch.save(cnn_class_imbalance.state_dict(),'wikiart_class_imbalance.pth')






