import numpy as np
import Linear_Regression

class KNN:

    def __init__(self, task, n_neighbors = 3, r = 2):

        self.n_neighbors = n_neighbors
        self.r = r
        self.task = task 

    
    def minkowski(self, X, y, x_sample):

        Distances = {}

        for i in range(X.shape[0]):

            Sum = 0

            for j in range(X.shape[1]):

                Sum += (X[i][j] - x_sample[j])** self.r
            
            Distances[i] = Sum ** (1/self.r)
        
        Sorted_Distances = dict(sorted(Distances.items(), key=lambda item: item[1]))

        k_nearest_neightbor = list(Sorted_Distances.keys())[:self.n_neighbors]

        k_nearest_neightbor_label = {}

        for indx in k_nearest_neightbor:

            k_nearest_neightbor_label[indx] = y[indx]

        if self.task == 'classification':

            Probable_labels = {}

            for item in list(k_nearest_neightbor_label.values()):

                Probable_labels[item] = list(k_nearest_neightbor_label.values()).count(item)
            
            Probable_labels = dict(sorted(Probable_labels.items(), key=lambda item: item[1]))

            return list(Probable_labels.keys())[-1]
        

        elif self.task == 'regression':
        
            Local_x = X[k_nearest_neightbor]
            Local_y = y[k_nearest_neightbor]

            LR = Linear_Regression.LinearRegression(Local_x,Local_y,learning_rate=0.1)#scaled_x,scaled_y)

            coef = LR.fit()
            print('x_sample',x_sample,type(x_sample),x_sample.shape,np.array(x_sample.reshape(1,3)))

            pre = LR.predict(x_sample)
            return pre
        


        else:

            raise ValueError('Only two tasks (classification / regression) are available.')
    

    def predict(self, X,y,x_test):

        Predicted = []

        for sample in range(x_test.shape[0]):

            Predicted.append(self.minkowski(X, y, x_test[sample]))
        

        return Predicted
    

    def accuracy(self, y_true, y_pred):

        if self.task == 'classification':

            return np.mean(y_true == y_pred)
        
        else:
            MAE = 0
            for sample in range(y_true.shape[0]):

                MAE += abs(y_true[sample] - y_pred[sample])
            
            MAE = MAE/(y_true.shape[0])

            return MAE





            



        """        elif self.task == 'regression':

            Sum = 0

            sumOfDist = 0

            for indx in k_nearest_neightbor:

                Sum +=   k_nearest_neightbor_label[indx] * (Sorted_Distances[indx])

                sumOfDist += Sorted_Distances[indx]
            

            return np.mean([k_nearest_neightbor_label[indx] for indx in k_nearest_neightbor])#Sum/sumOfDist
        """




"""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import KNearestNeighbors

# Generate binary classification data with three features
X, y = make_classification(n_samples=1000, n_features=3, n_informative=3,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Add more samples
additional_samples = 1000
X_additional, y_additional = make_classification(n_samples=additional_samples, n_features=3, n_informative=3,
                                                  n_redundant=0, n_clusters_per_class=1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])

# Split the dataset into training and testing sets
test_size = 0.3  # 20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


KNN = KNearestNeighbors.KNN(task = 'classification')

#KNN.fit(X_train,y_train)

y_pred = KNN.predict(X_train,y_train,X_test)

print(KNN.accuracy(y_test,y_pred))


-------------------------------
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import KNearestNeighbors
# Generate regression data with three features
X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)

# Add more samples
additional_samples = 1000
X_additional, y_additional = make_regression(n_samples=additional_samples, n_features=3, noise=0.1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])

# Split the dataset into training and testing sets
test_size = 0.3  # 30% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


KNN = KNearestNeighbors.KNN(task = 'regression',n_neighbors=10)

#KNN.fit(X_train,y_train)

y_pred = KNN.predict(X_train,y_train,X_test)

print(KNN.accuracy(y_test,y_pred))


import Ridge_Regression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
# Generate regression data with three features
X, y = make_regression(n_samples=1000, n_features=3, noise=0.1, random_state=42)

# Add more samples
additional_samples = 1000
X_additional, y_additional = make_regression(n_samples=additional_samples, n_features=3, noise=0.1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])
test_size = 0.3  # 30% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

rr = Ridge_Regression.RidgeRegression(X_train,y_train)

rr.fit()

y_pred=rr.predict(X_test)
print(rr.MAE(y_test,y_pred))




"""