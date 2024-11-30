import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score



class ClusteringAutoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(ClusteringAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim

        # Encoder (Convolutional Layers)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # 416x416 -> 208x208
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 208x208 -> 104x104
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 104x104 -> 52x52
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 52x52 -> 26x26
        self.relu4 = nn.ReLU()

        
        # Bottleneck (Latent Representation)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 512)   # Compress to 512 features
        self.fc2 = nn.Linear(512, self.latent_dim) # Compress to latent_dim (clustering space)
        
        # Decoder (Linear Layers)
        self.fc3 = nn.Linear(self.latent_dim, 512)          # Expand to 512 features
        self.fc4 = nn.Linear(512, 128 * 26 * 26)           # Expand to match encoder's feature map size
        
        # Decoder (Deconvolutional Layers)
        self.unflatten = nn.Unflatten(1, (128, 26, 26))
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding = 1, output_padding=1)   # 26x26 -> 52x52
        self.relu7 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding = 1, output_padding=1)    # 52x52 -> 104x104
        self.relu8 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding = 1, output_padding=1)    # 104x104 -> 208x208
        self.relu9 = nn.ReLU()
        self.deconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding = 1, output_padding=1)     # 208x208 -> 416x416
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        
        # Bottleneck
        x = self.flatten(x)
        x = self.fc1(x)
        latent = self.fc2(x)  # Clustering space
        
        # Decoder (Linear Layers)
        x = self.fc3(latent)
        x = self.fc4(x)
        
        # Decoder (Deconvolutional Layers)
        x = self.unflatten(x)
        x = self.deconv2(x)
        x = self.relu7(x)
        x = self.deconv3(x)
        x = self.relu8(x)
        x = self.deconv4(x)
        x = self.relu9(x)
        reconstructed = self.deconv5(x)
        reconstructed = self.sigmoid(reconstructed)
        
        return reconstructed, latent



def train_autoencoder_with_clustering(model, dataloader, epochs, device, latent_dim, k_clusters=10):
    """
    Train the autoencoder and evaluate clustering using true labels.
    
    Args:
        model (nn.Module): Autoencoder model.
        dataloader (DataLoader): DataLoader for training data.
        epochs (int): Number of training epochs.
        device (torch.device): Device (CPU or GPU).
        latent_dim (int): Dimension of the latent space.
        k_clusters (int): Number of clusters for K-Means (default: 10).
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        latent_representations = []
        all_labels = []

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = batch  # Assume labels are provided
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            reconstructed, latent = model(inputs)

            # Reconstruction loss
            loss = criterion(reconstructed, inputs)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss and latent vectors
            epoch_loss += loss.item()
            latent_representations.append(latent.detach().cpu())
            all_labels.append(labels.cpu())

        # Combine latent vectors and labels for clustering
        latent_representations = torch.cat(latent_representations)
        all_labels = torch.cat(all_labels).numpy()

        # Perform clustering and evaluate after a warm-up phase
        if epoch >= 0:  # Perform clustering after a few epochs
            kmeans = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
            cluster_assignments = kmeans.fit_predict(latent_representations.numpy())
            cluster_centers = kmeans.cluster_centers_

            # Evaluate clustering performance
            ari = adjusted_rand_score(all_labels, cluster_assignments)
            nmi = normalized_mutual_info_score(all_labels, cluster_assignments)

            print(f"Clustering Metrics at Epoch {epoch+1}:")
            print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
            print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")

        # Log epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    return model, cluster_centers

