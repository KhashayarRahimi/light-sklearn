import numpy as np
from Linear_Regression import LinearRegression


class RidgeRegression(LinearRegression):

    def __init__(self,x,y,iteration = 1000, learning_rate = 0.1,Lambda =1.0):

        super().__init__(x,y,iteration = 1000, learning_rate = 0.1)

        self.Lambda = Lambda
    
    #override on LinearRegression class
    def gradient(self,beta):

        x_t = np.transpose(self.x)

        x_beta = np.dot(self.x,beta)

        gradient = (-1/(self.x.shape[0])) * ( np.dot(x_t, self.y - x_beta) +\
        self.Lambda * beta)

        return gradient