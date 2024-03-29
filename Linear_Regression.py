import numpy as np

class LinearRegression:

    def __init__(self,x,y,iteration = 1000, learning_rate = 0.1):
        
        # input matrix
        self.x = np.c_[ x, np.ones(x.shape[0]) ]

        # output vector
        self.y = y

        # number of iteration for converging to optimal value (minimum for loss function)
        self.iteration = iteration

        self.learning_rate = learning_rate

        # default value of beta which is the weights or coefficient


    
    def gradient(self,beta):

        x_t = np.transpose(self.x)

        x_beta = np.dot(self.x,beta)

        gradient = (-1/(self.x.shape[0])) *  np.dot(x_t, self.y - x_beta)

        return gradient
    
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
        #print('X_test',X_test)

        Coefs = self.fit(Coef = False)

        y_pred = np.dot(X_test,Coefs)

        return y_pred
    
    def MAE(self, y_true, y_pred):
        

        MAE = 0
        for sample in range(y_true.shape[0]):

            MAE += abs(y_true[sample] - y_pred[sample])
        
        MAE = MAE/(y_true.shape[0])

        return MAE
        


#----------------------------------------

"""
Examples:
------------------------------------------------

from sklearn.metrics import mean_absolute_error
import Linear_Regression
import numpy as np
import random


X = []

for i, j , k in zip(range(100000),range(100000), range(100000)):

    X.append([random.random(),random.random(),random.random()])

X = np.array(X)

y = np.dot(X, np.array([1.5+random.random(), -2.7+random.random() , .01+random.random()]))+3


LR = Linear_Regression.LinearRegression(s,y,learning_rate=0.1)#scaled_x,scaled_y)

coef = LR.fit()

print(coef) ---> [ 2.22079927 -1.78341178  0.80279153  2.99049536]

test = np.array([[1,2,3],[22,0,2]])
LR.predict(test) ---> array([-0.27789246, 40.00380568])
print(LR.MAE(y,y_pred)) ---> 0.00294


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
print(reg.coef_,reg.intercept_)  ---> [ 2.21527174 -1.78997526  0.79681398] 2.999999999999995
reg.predict(test) ---> array([-0.32091392, 39.80032898])
print(mean_absolute_error(y, y_pred)) ---> 0.00294
"""