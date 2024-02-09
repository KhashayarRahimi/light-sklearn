import numpy as np
from Linear_Regression import LinearRegression


class LassoRegression(LinearRegression):

    def __init__(self,x,y,iteration = 1000, learning_rate = 0.1,Lambda =1.0):

        super().__init__(x,y,iteration, learning_rate)

        self.Lambda = Lambda
        

    
    def _vetor_abs(self, vector):

        for indx in range(len(vector)):

            if vector[indx] < 0:

                vector[indx] = abs(vector[indx])
        
        return vector


    
    #override on LinearRegression class
    def gradient(self,beta):

        x_t = np.transpose(self.x)

        x_beta = np.dot(self.x,beta)

        absOfWeights = self._vetor_abs(beta)

        gradient = (-1/(self.x.shape[0])) * ( np.dot(x_t, self.y - x_beta) + self.Lambda * absOfWeights)

        return gradient


#----------------------------------------

"""
Examples:
------------------------------------------------
import random
import numpy as np
import Linear_Regression
import Lasso_Regression
from sklearn.metrics import mean_absolute_error


X = []
f1 = [random.random() for i in range(5000)]

for f in range(1,10):
    X.append([f*i for i in f1])

X = np.transpose(np.array(X))
Real_Weights = np.array([((-1)**f) * random.random() for f in range(1,10)])
y = np.dot(s, Real_Weights)+3

LR = Linear_Regression.LinearRegression(X,y,
                                        iteration = 1000,
                                        learning_rate=0.01)

coef = LR.fit()
y_pred = LR.predict(X)
print(LR.MAE(y,y_pred)) ---> 0.1034

LaR = Lasso_Regression.LassoRegression(X,y,iteration = 1000,
                                       learning_rate=0.01,
                                       Lambda=135.0)

coef = LaR.fit()
y_pred = LaR.predict(X)
print(LaR.MAE(y,y_pred)) ---> 0.0046
"""