## 1. Improving Performance of the Old Model

Uses the files `improved_train.py` and `test.py` 


**Issue with the Previous Model:**

The previous model lacked **non-linearity** after the convolutional layer and was only appled after the linear layer. 

Improvements Implemented:
   - Added two further convolutional layer (taken from the Thai OCR model)
   - Added activation functions (e.g., ReLU) after each convolutional layer for more non-linearity.                                                  
   - Used a very low learning rate and implemented a **very rigorous** early stopping monitor.

**Improved accuracy is now at 8.4 percent!**

## 2. Strategies to Handle Class Imbalance:

Uses the files `class_imbalance.py` and `test.py`.

1. **Measure 1: Data Augmentation for Minority Classes**
   - I used data augmentation to create additional, varied examples of the minority classes.
   - I implemented horizontal flipping, random rotation, and cropping.

2. **Measure 2: Class Weights in the Loss Function**
   - I use class weights adjusted to the loss function to give more importance to minority classes. 
   - Class weights are calculated inversely proportional to the class frequencies using `compute_class_weight` and are assigned to the loss function.

3. **Measure 3: Oversampling Minority Classes**
   - I oversampled the minority class samples to ensure a balanced number of samples for all classes. 
     - A `WeightedRandomSampler` is used to generate a balanced batch by assigning higher sampling weights to minority class instances.

4. **Measure 4: Use of Focal Loss**
     - I also tried to implement a focal loss function, which you can stil find  as an option to select in the model constitution, but it consequently lead to overflow. 
     - As the overflow still occured after several regulatory measures (like gradient clipping) I abandoned this attempt.  

## 3. Clustering Latent Representations

Uses the files `autoencoder.py` and `train_autoencoder.py`

1. **Extracting Latent Representations**
   - I use an autoencoder that combines a convolutional and a linear layer to compress the model to 32 dimensions. See `autoencoder.py`.

2. **Clustering Latent Representations**
   - **K-Means Clustering** is applied to the latent vectors instead of other clustering methods like density based DBSCAN because the number of clusters is known. 

3. **Evaluating Clustering**
   - **Silhouette Score** is calculated to measure the quality of clustering:
   - A higher silhouette score indicates that points are well-separated from other clusters and closer to their own cluster center.

4. **Visualizing the Latent Space**
   - Latent vectors are reduced to 2D using **t-SNE**, which works better for visualizing high-dimensional data in two dimensions than PCA, especially for complex, non-linear data like images.
   - **Scatter Plots**:
   - The first plot colors data points using true labels, showing how well latent representations align with ground truth.
   - The second plot uses K-Means cluster assignments to visualize how the data was grouped.
