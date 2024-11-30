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


trainingdir = '/scratch/lt2326-2926-h24/wikiart/train'

device = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = torch.cuda.current_device()


print("Running...")


dataset = WikiArtDataset(trainingdir, device)

print(dataset.imgdir)



labels = [label for _, label in dataset]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in split.split(dataset, labels):
    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)

the_image, the_label = val_data[5]
print(the_image, the_image.size())
the_showable_image = F.to_pil_image(the_image)
print("Label of img 5 is {}".format(the_label))
the_showable_image.show()
shape = the_image.shape[1]


num_of_labels = len(dataset.classes)
cnn_model = CNNWikiArt(300, output_size=num_of_labels, optimizer = 'SGD', learning_rate = 0.0001, l2 = 0.0).to(device)
print(next(cnn_model.parameters()).device)
trainer = Trainer(max_epochs = 15, batch_size = 32, early_stopping_patience = 2, min_delta = 0.009 )
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

#################################################################


torch.save(cnn_model.state_dict(),'wikiart.pth')


