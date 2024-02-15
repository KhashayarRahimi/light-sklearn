import numpy as np

class LinearSVC:

    def __init__(self, learning_rate = 0.01, Lambda = 0.01, iteration = 1000,task='classification'):

        self.learning_rate = learning_rate
        self.Lambda = Lambda
        self.iteration = iteration
        self.w = None
        self.b = None
        self.x = None
        self.y = None
        self.task = task
    

    def fit(self, x, y):

        self.x = x
        self.y = y

        self.w = np.zeros(x.shape[1])
        self.b = 0

        for iter in range(self.iteration):
        
            for indx in range(x.shape[0]):

                if self.y[indx] * ((np.dot(self.x[indx], self.w)) - self.b) >= 1:

                    self.w = self.w - self.learning_rate * (2 * self.Lambda * self.w)

                else:

                    self.w = self.w - self.learning_rate * (2 * self.Lambda * self.w - np.dot(self.x[indx], self.y[indx]))

                    self.b = self.b -  self.learning_rate * self.y[indx]
        
        return self.w , self.b
    
    def predict(self, x_test):

        w, b = self.fit(self.x, self.y)

        y_pred = np.dot(x_test, w) - b

        y_pred_class = [1 if val > 0 else -1 for val in y_pred]

        return y_pred_class,y_pred, w,b
    
    def accuracy(self, y_true, y_pred):

        if self.task == 'classification':

            return np.mean(y_true == y_pred)
        
        else:
            MAE = 0
            for sample in range(y_true.shape[0]):

                MAE += abs(y_true[sample] - y_pred[sample])
            
            MAE = MAE/(y_true.shape[0])

            return MAE


#-------------------------------------------------------------
"""
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy as np
import re
import Support_Vector_Machine


X, y = make_classification(n_samples=1000, n_features=3, n_informative=3,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Add more samples

X_additional, y_additional = make_classification(n_samples=1000, n_features=3, n_informative=3,
                                                  n_redundant=0, n_clusters_per_class=1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])
y = np.where(y==1,y,-1)

x_scaler = StandardScaler()
scaled_x = x_scaler.fit_transform(X).reshape(X.shape)
#y_scaler = StandardScaler()
#scaled_y = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

# Split the dataset into training and testing sets
test_size = 0.3  # 20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=test_size, random_state=42)

LSVC = Support_Vector_Machine.LinearSVC(learning_rate=0.001,iteration=1000,Lambda=0.1)

LSVC.fit(X_train,y_train)

y_pred = LSVC.predict(X_test)

print(LSVC.accuracy(y_test,y_pred[0])) ---> 0.925
"""




