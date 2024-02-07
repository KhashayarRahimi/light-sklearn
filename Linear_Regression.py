import numpy as np

class LinearRegression:

    def __init__(self,x,y,iteration = 100, learning_rate = 0.1):
        
        # input matrix
        self.x = x

        # output vector
        self.y = y

        # number of iteration for converging to optimal value (minimum for loss function)
        self.iteration = iteration

        self.learning_rate = learning_rate

        # default value of beta which is the weights or coefficient
        #self.beta = np.zeros(x.shape[1])

    
    def gradient(self,beta):

        x_t = np.transpose(self.x)

        x_beta = np.dot(self.x,beta)

        gradient = -1 *  np.dot(x_t, self.y - x_beta)

        return gradient
    
    def update(self, beta):

        new_beta = beta - self.learning_rate * self.gradient(beta)

        return new_beta

        
    def fit(self):

        beta = np.zeros(self.x.shape[1])

        for iter in range(self.iteration):

            beta = self.update(beta)

        return beta
        

    def predict():
        pass

    def coefficient(): # weights+intercept
        pass
