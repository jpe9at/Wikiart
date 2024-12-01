import torch 
import torch.nn as nn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os
from autoencoder import ClusteringAutoencoder, train_autoencoder_with_clustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from wikiart import WikiArtDataset
from torch.utils.data import DataLoader

import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Determine number of epochs.")
parser.add_argument("--number", type=int, default = 10, help="The number to add.")
parser.add_argument("--device", type=int, default = 0, help="The device number.")

args = parser.parse_args()
number_of_epochs = args.number

device = args.device
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = torch.cuda.current_device()


trainingdir = '/scratch/lt2326-2926-h24/wikiart/train'
traindataset = WikiArtDataset(trainingdir, device)

latent_dim = 32  # Latent space dimensionality for clustering
k_clusters = 27  # Number of clusters

# Initialize the model
wiki_art_autoencoder = ClusteringAutoencoder(latent_dim=latent_dim)
dataloader = DataLoader(traindataset, batch_size=64, shuffle = True)

# Train the model, this time simplified without validation set
wiki_art_autoencoder, cluster_centers = train_autoencoder_with_clustering(
    model=wiki_art_autoencoder,
    dataloader=dataloader,
    epochs=number_of_epochs,
    device=device,
    latent_dim=latent_dim,
    k_clusters=k_clusters
)


torch.save(wiki_art_autoencoder.state_dict(),'wikiart_autoencoder.pth')


# Get a list of all latent vectors:
latent_vectors = []
all_labels = []
wiki_art_autoencoder.eval()
with torch.no_grad():
    for inputs, labels in dataloader:
        all_labels.append(labels.cpu())
        inputs = inputs.to(device)
        _, latent = wiki_art_autoencoder(inputs)
        latent_vectors.append(latent.cpu())

latent_vectors = torch.cat(latent_vectors)
all_labels = torch.cat(all_labels).numpy()

# Perform clustering
# Kmeans is chosen and not methods like DBSCAN because the number of clusters is known. 
kmeans = KMeans(n_clusters=k_clusters)
cluster_assignments = kmeans.fit_predict(latent_vectors.numpy())

# Evaluate clustering since labels are available)
silhouette = silhouette_score(latent_vectors.numpy(), cluster_assignments)
print(f"Silhouette Score: {silhouette}")

# Reduce latent vectors to 2D for visualization
latent_2d = TSNE(n_components=2, random_state=42).fit_transform(latent_vectors.numpy())

# Scatter plot with true labels
plt.figure(figsize=(10, 6))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=all_labels, cmap='viridis', s=5)
plt.colorbar(label='True Labels')
plt.title('Latent Space Visualization with True Labels')
plt.savefig('true_labels.png')

# Scatter plot with cluster assignments
plt.figure(figsize=(10, 6))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_assignments, cmap='viridis', s=5)
plt.colorbar(label='Cluster Assignments')
plt.title('Latent Space Visualization with Cluster Assignments')
plt.savefig('cluster_assignments.png')
