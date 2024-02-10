import numpy as np
import math

class BinaryLogisticRegression:

    def __init__(self, x, y, iteration = 1000, learning_rate = 0.1):

        self.x = np.c_[ x, np.ones(x.shape[0]) ]
        self.y = y
        self.iteration = iteration
        self.learning_rate = learning_rate
    

    def sigmoid(self, vector):

        new_vec = []

        for i in vector:

            #print(abs(i))

            new_vec.append(1/(1+math.exp(-i)))

        return new_vec
    
    def gradient(self, beta):

        a = self.sigmoid(np.dot(self.x, beta)) #in Murphy book is mu_n

        negative_log_likelihood_gradient = (1/(self.x.shape[0])) * np.dot((a - self.y) , self.x)

        return negative_log_likelihood_gradient
    
    
    def update(self, beta):

        new_beta = beta - self.learning_rate * self.gradient(beta)

        return new_beta

        
    def fit(self,Coef = True):

        beta = np.zeros(self.x.shape[1])

        for iter in range(self.iteration):

            beta = self.update(beta)

        if Coef:

            print(f"Coefficients:{beta[:-1]} \n Intercept:{beta[-1]}")

        return beta

    
    
    def predict(self,X_test):

        X_test = np.c_[ X_test, np.ones(X_test.shape[0]) ]

        Coefs = self.fit(Coef = False)

        y_pred = np.dot(X_test,Coefs)

        Prediction = []

        for y in y_pred:
            #print(y)

            prob = math.exp(y) / (1+math.exp(y))

            if prob >= 0.5:


                Prediction.append(1)
            
            else:
                Prediction.append(0)


        return Prediction
    
    
    def accuracy(self, y_true, y_pred):

        right_prediction = 0

        for i in range(len(y_true)):

            if y_true[i] == y_pred[i]:

                right_prediction += 1


        acc = right_prediction / len(y_pred)

        return acc
    

#----------------------------------------

"""
Examples:
------------------------------------------------
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np 

# Generate binary classification data with three features
X, y = make_classification(n_samples=1000, n_features=3, n_informative=3,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Add more samples
additional_samples = 1000
X_additional, y_additional = make_classification(n_samples=additional_samples, n_features=n_features, n_informative=3,
                                                  n_redundant=0, n_clusters_per_class=1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])

# Split the dataset into training and testing sets
test_size = 0.3  # 20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

LogR = Logistic_Regression.BinaryLogisticRegression(X_train,y_train)

LogR.fit()

y_pred = LogR.predict(X_test)

print(LogR.accuracy(y_test,y_pred)) ---> 0.923

""" 
    

class MultiClassLoisticRegression(BinaryLogisticRegression):

    def __init__(self, x, y, iteration = 1000, learning_rate = 0.1):

        super().__init__(x,y,iteration, learning_rate)

        self.x = np.c_[ x, np.ones(x.shape[0]) ]

        self.num_cat = len(set(self.y))
        self.negative_log_likelihood_gradient = np.zeros((len(set(self.y)),self.x.shape[1]))
    
    
    def softamx(self, cat, Beta, row):

        SUM = 0
        #row = np.c_[ row, np.ones(1) ]

        for cat_ in range(self.num_cat):
            #print(row,np.transpose(beta))

            SUM += math.exp(np.dot(np.transpose(Beta[cat_]),row))#math.exp(np.dot(row,np.transpose(beta)))
        
        cat_prob =math.exp(np.dot(np.transpose(Beta[cat]),row))/SUM #math.exp(np.dot(row,np.transpose(Beta[cat]))) / SUM

        return cat_prob
    
    def gradient_matrix(self, Beta):

        for cat in range(self.num_cat):

            SumOnN = 0

            for sample_indx in range(self.x.shape[0]):

                if self.y[sample_indx] == cat:
                    y_ik = 1
                else:
                    y_ik = 0
                

                SumOnN += (y_ik - self.softamx(cat, Beta, self.x[sample_indx])) * np.transpose(self.x[sample_indx])
            
            self.negative_log_likelihood_gradient[cat] = (-1/self.x.shape[0]) * SumOnN

        return self.negative_log_likelihood_gradient
    

    def update(self, Beta):

        new_Beta = Beta - self.learning_rate * self.gradient_matrix(Beta)

        return new_Beta
    

    def fit(self,Coef = True):

        Beta = np.zeros((self.num_cat,self.x.shape[1]))


        for iter in range(self.iteration):

            Beta = self.update(Beta)

        if Coef:

            print(f"Coefficients and Intercept: \n {Beta}")

        return Beta
    

    def predict(self,X_test):

        X_test = np.c_[ X_test, np.ones(X_test.shape[0]) ]

        Coefs = self.fit(Coef = False)
        
        
        Predicted_label = []
        linearCombinations = np.dot(X_test,np.transpose(Coefs))
        
        for row in linearCombinations:

            catProb = {}

            denominator = [math.exp(row[cat]) for cat in range(self.num_cat)]

            for cat in range(self.num_cat):

                catProb[math.exp(row[cat])/sum(denominator)] = cat
            
            Predicted_label.append(catProb[max(catProb.keys())])
        

        return Predicted_label
    
    
    def accuracy(self, y_true, y_pred):

        right_prediction = 0

        for i in range(len(y_true)):

            if y_true[i] == y_pred[i]:

                right_prediction += 1


        acc = right_prediction / len(y_pred)

        return acc


#----------------------------------------

"""
Examples:
------------------------------------------------
import Logistic_Regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np 

# Generate data with three class labels and three features
X, y = make_classification(n_samples=500, n_features=6, n_informative=6,
                           n_redundant=0, n_classes=3, random_state=42)

# Add more samples
additional_samples = 500
X_additional, y_additional = make_classification(n_samples=additional_samples, n_features=6,
                                                  n_informative=6,
                                                  n_redundant=0, n_classes=3, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])

# Split the dataset into training and testing sets
test_size = 0.3  # 30% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


LogR = Logistic_Regression.MultiClassLoisticRegression(X_train,y_train,iteration= 100,
                                                       learning_rate=0.01)

LogR.fit()

y_pred = LogR.predict(X_test)

print(LogR.accuracy(y_test,y_pred)) ---> 0.7133

-------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a logistic regression model
model = LogisticRegression(max_iter=500, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set

y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) ---> Accuracy: 0.71

""" 