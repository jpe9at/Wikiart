import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, CNNWikiArt
import json
import argparse

import numpy as np

from TrainerWiki import Trainer
from sklearn.model_selection import StratifiedShuffleSplit


parser = argparse.ArgumentParser(description="Determine number of epochs.")
parser.add_argument("--number", type=int, default = 10, help="The number to add.")
parser.add_argument("--device", type=int, default = 0, help="The device number.")

args = parser.parse_args()
number_of_epochs = args.number




device = args.device
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = torch.cuda.current_device()


print("Running...")

# Get training and validation data
trainingdir = '/scratch/lt2326-2926-h24/wikiart/train'
dataset = WikiArtDataset(trainingdir, device)
print(dataset.imgdir)

labels = [label for _, label in dataset]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(dataset, labels):
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)


# Initialise CNN ad train
num_of_labels = len(dataset.classes)
cnn_model = CNNWikiArt(300, output_size=num_of_labels, optimizer = 'SGD', learning_rate = 0.00006, l2 = 0.001).to(device)
print(next(cnn_model.parameters()).device)
trainer = Trainer(max_epochs = number_of_epochs, batch_size = 32, early_stopping_patience = 2, min_delta = 0.0009 )
trainer.fit(cnn_model,train_data,val_data)


#Plot Progress report
n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss,nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss,nan_values])

plt.figure(figsize=(10,6))
plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
plt.title("Train Loss")
plt.legend()
plt.savefig("Improved_Train_Val_loss.png")

# Save the model
torch.save(cnn_model.state_dict(),'wikiart.pth')


