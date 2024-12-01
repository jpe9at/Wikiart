import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset,  CNNWikiArt
import torcheval.metrics as metrics
import json
import argparse

import seaborn as sns

##################################################
# This file works with models 
#     wikiart.pth and 
#     wikiart_class_imbalance.pth
# uncomment the respective line where the function is called.
#################################################



parser = argparse.ArgumentParser(description="Determine number of epochs.")
parser.add_argument("--device", type=int, default = 0, help="The device number.")

args = parser.parse_args()

device = args.device
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = torch.cuda.current_device()

print("Running...")

testingdir = '/scratch/lt2326-2926-h24/wikiart/test'
testingdataset = WikiArtDataset(testingdir, device)


def test(modelfile=None, device="cpu"):
    loader = DataLoader(testingdataset, batch_size=1)

    model =  CNNWikiArt(300, output_size=27)
    model.load_state_dict(torch.load(modelfile, weights_only=True), strict = False)
    model = model.to(device)
    model.eval()

    predictions = []
    truth = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        y = y.to(device)
        output = model(X)
        predictions.append(torch.argmax(output).unsqueeze(dim=0))
        truth.append(y)

    predictions = torch.concat(predictions)
    truth = torch.concat(truth)
    metric = metrics.MulticlassAccuracy()
    metric.update(predictions, truth)
    print("Accuracy: {}".format(metric.compute()))
    confusion = metrics.MulticlassConfusionMatrix(27)
    confusion.update(predictions, truth)
    confusion_matrix = confusion.compute().cpu().numpy()
    return confusion_matrix

#confusion_matrix = test(modelfile='wikiart.pth', device=device)

confusion_matrix = test(modelfile='wikiart_class_imbalance.pth', device=device)

plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
plt.savefig("confusion_matrix.png")