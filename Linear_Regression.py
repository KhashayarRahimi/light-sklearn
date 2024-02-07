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
        self.beta = np.zeros(x.shape[1])

    def update(self, updated_beta):

        return 1

    
    def gradient(self,x,y,beta):

        x_t = np.transpose(self.x)

        x_beta = np.dot(self.x,beta)

        gradient = -2 *  np.dot(x_t, self.y - x_beta)

        return gradient

        



    def fit(self):

        #compute the gradient
        return 1
        

    def predict():
        pass
