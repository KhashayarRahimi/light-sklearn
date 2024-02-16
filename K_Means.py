import numpy as np
import random

class Kmeans:

    def __init__(self ,minkowski_degree = 2, n_clusters = 3):

        self.minkowski_degree = minkowski_degree
        self.n_clusters = n_clusters
    
    def minkowskiDistance(self, sample1, sample2):

        Sum = 0

        if sample1.shape != sample2.shape:

            raise ValueError(f'{sample1.shape} != {sample2.shape}')

        for i in range(sample1.shape[0]):

            Sum += (sample1[i] - sample2[i]) ** self.minkowski_degree
        
        return Sum
    
    def _randomIndexDedication(self, X):

        RandomClusters = [i for i in range(self.n_clusters)]

        DedicatedRandomClusters = []

        for i in range(X.shape[0]):

            DedicatedRandomClusters.append(random.choice(RandomClusters))
        
        return DedicatedRandomClusters
    
    def _WithinClustersMean(self, X, NewCluster):

        ClustersVariation = {}

        for cluster in range(self.n_clusters):

            indices = np.where(np.array(NewCluster) == cluster)

            ClusterMean = np.mean(X[indices], axis=0)

            ClustersVariation[cluster] = np.array([round(i,3) for i in ClusterMean])
        
        return ClustersVariation
    
    def fit(self, X):

        new_clustering = self._randomIndexDedication(X)

        Change_indicator = 0

        while Change_indicator != 'Done':

            BeforeClustersMean = self._WithinClustersMean(X, new_clustering)
            #print(BeforeClustersMean)

            for i in range(X.shape[0]):

                Sampel_i_distance = {}

                for cluster in list(BeforeClustersMean.keys()):

                    #print(ClustersMean[cluster], X[i])
                    Sampel_i_distance[round(self.minkowskiDistance(BeforeClustersMean[cluster], X[i]),3)] = cluster
                #print(i,Sampel_i_distance)
                
                BetterCluster = Sampel_i_distance[min(list(Sampel_i_distance.keys()))]

                if BetterCluster != new_clustering[i]:

                    Change_indicator += 1

                    new_clustering[i] = BetterCluster
            
            AfterClustersMean = self._WithinClustersMean(X, new_clustering)

            equal = True
            for key in BeforeClustersMean.keys():
                if not np.array_equal(BeforeClustersMean[key], AfterClustersMean[key]):
                    equal = False
                    break

            if equal:
                Change_indicator = 'Done'
                    

        return new_clustering, AfterClustersMean


    def predict(self, FinalClustersMean, X_test):

        prediction = []

        for i in range(X_test.shape[0]):

            Sampel_i_distance = {}

            for cluster in list(FinalClustersMean.keys()):

                Sampel_i_distance[round(self.minkowskiDistance(FinalClustersMean[cluster], X_test[i]),3)] = cluster

            
            BetterCluster = Sampel_i_distance[min(list(Sampel_i_distance.keys()))]

            prediction.append(BetterCluster)
        
        return prediction

#-----------------------------------------------------------------------------------

"""
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import K_Means


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

kmean = K_Means.Kmeans()

clustering = kmean.fit(X_train)

plt.scatter(X_train[:,0], X_train[:,1], c=clustering[0], cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

prediction = kmean.predict(clustering[1], X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=prediction, cmap='viridis', marker='o', s=30)
plt.title('Generated Data with Cluster Labels')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
"""