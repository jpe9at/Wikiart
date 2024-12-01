For the programs `improved_train.py`, `class_imbalance.py`, and `train_autoencoder.py` you can determine the number of epochs by the optional command 

--number 

and the number of the device by the optional command 

--device


## 1. Improving Performance of the Old Model

Uses the files `improved_train.py` and `test.py` 


**Issue with the Previous Model:**

The previous model lacked **non-linearity** after the convolutional layer and was only appled after the linear layer. 

Improvements Implemented:
   - Added two further convolutional layer (taken from the Thai OCR model)
   - Added activation functions (e.g., ReLU) after each convolutional layer for more non-linearity.                                                  
   - Attempt one: Used SGD and a very low learning rate (0.0001) and implemented a **very rigorous** early stopping monitor (stopping after one epoch of no improvements), plus no l2 regularisation. Ended up only training for one epoch, but: Improved accuracy is now at **8.4 percent!** But since training for one epoch can hardly be called training, I started another attempt.  
   - Attempt two: Used SGD and an even lower learning rate of 0.00006 and l2 regularisation of 0.001. Trained for 20 epochs, but convergence was not yet reacherd. Improved accuracy is now at **5.8 percent!**

## 2. Strategies to Handle Class Imbalance:

Uses the files `class_imbalance.py` and `test.py`.

1. **Measure 1: Data Augmentation for Minority Classes**
   - I used data augmentation to create additional, varied examples of the minority classes.
   - I implemented horizontal flipping, random rotation, and cropping.
   - As part of this, the `WikiartDataset` class was augmented with an optional argument, which takes a list of minority classes and applied the data augmentation to those when calling the `__getitem__` method. 

2. **Measure 2: Class Weights in the Loss Function**
   - I use class weights adjusted to the loss function to give more importance to minority classes. 
   - Class weights are calculated inversely proportional to the class frequencies using `compute_class_weight` and are assigned to the loss function.

3. **Measure 3: Oversampling Minority Classes**
   - I oversampled the minority class samples to ensure a balanced number of samples for all classes. 
     - A `WeightedRandomSampler` is used to generate a balanced batch by assigning higher sampling weights to minority class instances.

4. **Measure 4: Use of Focal Loss**
   - I also tried to implement a focal loss function. The **Focal Loss** for the $i$-th sample is given by:

       $\text{FL}_i = -\alpha (1 - p_{i})^\gamma \log(p_{i}),$

      where:
     - $p_{i}$ is the probability of sample $i$ for the true label
     - $\alpha$: A weighting factor for class imbalance.
     - $\gamma$: A focusing parameter that down-weights easy examples ($p_{t,i}$ close to 1).

   - As overflow occurred and still occured after several regulatory measures (like clamping the loss function and using gradient clipping) I abandoned this attempt.  

## 3. Clustering Latent Representations

Uses the files `autoencoder.py` and `train_autoencoder.py`

1. **Extracting Latent Representations**
   - I use an autoencoder that combines convolutional and linear layers to compress the model to 32 dimensions. See `autoencoder.py`.
   - Used MSE as a loss function, because the error is a continous difference between the reconstructed image and the original. 

2. **Clustering Latent Representations**
   - **K-Means Clustering** is applied to the latent vectors instead of other clustering methods like density based DBSCAN because the number of clusters is known. 

3. **Evaluating Clustering**
   - **Silhouette Score** is calculated to measure the quality of clustering.
   - For a single data point $i$:

      - $a(i)$: The average distance of $i$ to all other points in the same cluster (intra-cluster distance).
      - $b(i)$: The average distance of $i$ to all points in the nearest neighboring cluster (inter-cluster distance).

      The silhouette score for point $i$ is defined as: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$,
      where $s(i)$ ranges from -1 to 1.

   - The overall silhouette score for a clustering solution is the mean $s(i)$ across all data points. A higher silhouette score indicates that points are well-separated from other clusters and closer to their own cluster center. Final value: 0.5183, which means there is moderate cluster building, but also some overlap between the clusters. 

4. **Visualizing the Latent Space**
   - Latent vectors are reduced to 2D using **t-SNE**, which works better for visualizing high-dimensional data in two dimensions than PCA, especially for complex, non-linear data like images.
   - **Scatter Plots**:
   - The first plot colors data points using true labels, showing how well latent representations align with ground truth.
   - The second plot uses K-Means cluster assignments to visualize how the data was grouped.
