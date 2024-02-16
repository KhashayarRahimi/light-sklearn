"""
The implementation of this class is belong the following website:
https://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
However I have modified it a little;
1. initializer using kmeans
2. X_test in predict and predict_proba method instead of X and
likelihood = np.zeros( (X_test.shape[0], self.k) )
instead of
likelihood = np.zeros( (self.n, self.k) )
"""
import numpy as np
from scipy.stats import multivariate_normal
import K_Means

class GMM:
    def __init__(self, k, max_iter=100, intializer = 'random'):
        self.k = k
        self.max_iter = int(max_iter)
        self.intializer = intializer

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full( shape=self.shape, fill_value=1/self.k)
        
        if self.intializer == 'random':
            random_row = np.random.randint(low=0, high=self.n, size=self.k)
            self.mu = [  X[row_index,:] for row_index in random_row ]

        elif self.intializer == 'kmeans':
            kmean = K_Means.Kmeans().fit(X)

            self.mu = kmean[1]
        #
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ]

    def e_step(self, X):
        # E-Step: update weights and phi holding mu and sigma constant
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)
    
    def m_step(self, X):
        # M-Step: update mu and sigma holding phi and weights constant
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)

    def fit(self, X):
        self.initialize(X)
        
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            
    def predict_proba(self, X_test):
        likelihood = np.zeros( (X_test.shape[0], self.k) )
        for i in range(self.k):
            distribution = multivariate_normal(
                mean=self.mu[i], 
                cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X_test)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X_test):
        weights = self.predict_proba(X_test)
        return np.argmax(weights, axis=1)
    
#------------------------------------
    
"""
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import Gaussian_Mixture


def generate_data(num_samples, num_features, num_clusters):

    # Generate random cluster centroids
    centroids = np.random.rand(num_clusters, num_features) * 10
    
    # Generate data points around the centroids
    data = []
    labels = []
    for i in range(num_clusters):
        cluster_data = np.random.randn(num_samples//num_clusters, num_features) + centroids[i]
        data.append(cluster_data)
        labels.extend([i] * (num_samples//num_clusters))
    
    data = np.concatenate(data, axis=0)
    labels = np.array(labels)
    
    # Shuffle data and labels together
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    
    return data, labels

# Example usage:
num_samples = 3000
num_features = 2
num_clusters = 3

data, labels = generate_data(num_samples, num_features, num_clusters)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Plot the generated data with cluster labels
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

GMM = Gaussian_Mixture.GMM(k = 3, max_iter=100)
GMM.fit(X_train)


plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

prediction = GMM.predict(X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=prediction, cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
"""