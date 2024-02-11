import numpy as np
import math

class LinearDiscriminantAnalysis:

    def __init__(self, x, y):

        self.x = x
        self.y = y
    
    def mean_vector(self, cat):

        indices = np.where(self.y == cat)[0]
        # Calculate mean for each feature for samples belonging to the current target
        mean_per_feature = np.mean(self.x[indices], axis=0)

        return mean_per_feature
    
    def covariance(self,var1, var2):

        #print(var1,var2)
        #print('------')

        sum = 0

        if True: #var1 != var2:

            for i in range(len(var1)):

                sum += (var1[i] - np.mean(var1)) * (var2[i] - np.mean(var2))
                #print(sum)
        else:
            
            for i in range(len(var1)):

                sum += (var1[i] - np.mean(var1)) ** 2
        
        return sum/(len(var1)-1)
    
    # Note that the elements in args must be the columns of the data; arrays
    def covariance_matrix(self, args):

        args = np.transpose(args)

        # dimension of the covariance matrix
        CovMatrix = np.zeros((args.shape[0],args.shape[0]))
        #print(CovMatrix.shape)
        #print(args[0])

        for var1 in range(args.shape[0]):

            for var2 in range(args.shape[0]):
                #print(var1,var2)
                #print('---')
                #print(args[var1],args[var2])

                VarCov = self.covariance(args[var1],args[var2])
                #print('&&',VarCov)

                CovMatrix[var1][var2] = VarCov
        
        return CovMatrix
    
    def Pi_cat(self,cat):

        return len(np.where(self.y == cat)[0])/self.y.shape[0]
    

    def delta_freeX(self,cat,InvCovMatrix):


        Mu_cat = self.mean_vector(cat)
        #print(Mu_cat)
        #print('---')
        #print(InvCovMatrix.shape)
        CovInvMuCat = np.dot(InvCovMatrix,Mu_cat)

        term1 = CovInvMuCat #np.dot(sample_x,CovInvMuCat)
        term2 = (-1/2) * np.dot(np.transpose(Mu_cat),CovInvMuCat)
        term3 = math.log(self.Pi_cat(cat))
        #print(self.Pi_cat(cat),cat)

        return term1, term2+term3
    
    def fit(self):

        CovMatrix = {}
        InvCovMatrix = {}

        for cat in range(len(set(self.y))):

            CovMatrix[cat] = self.covariance_matrix(self.x[np.where(self.y == cat)[0]])
        #print(CovMatrix)
        

            try:
                InvCovMatrix[cat] = np.linalg.inv(CovMatrix[cat])

            # The Covariance matrix is singular and does not have invers;
            # therefore we add a small value to its diagonal
            except np.linalg.LinAlgError:

                np.fill_diagonal(CovMatrix[cat], CovMatrix[cat].diagonal() + 0.0001)
                InvCovMatrix[cat] = np.linalg.inv(CovMatrix[cat])

        #print('CovMatrix',CovMatrix)
        #print('InvCovMatrix',InvCovMatrix)
        Information = {}

        for cat in range(len(set(self.y))):

            D = self.delta_freeX(cat, InvCovMatrix[cat])

            Information[cat] = [D[0], D[1]]
        
        return Information
    
    def predict(self, x_test):

        Information = self.fit()#self.x, self.y)

        Prediction = []
        for sample in x_test:

            Predict_labels = {}

            for cat in range(len(set(self.y))):
            
                Predict_labels = {}

                delta_cat_x = np.dot(sample, Information[cat][0]) + Information[cat][1]

                Predict_labels[delta_cat_x] = cat

            Prediction.append(Predict_labels[max(Predict_labels.keys())])
        
        return Prediction
    
    
    def accuracy(self, y_true, y_pred):

        right_prediction = 0

        for i in range(len(y_true)):

            if y_true[i] == y_pred[i]:

                right_prediction += 1


        acc = right_prediction / len(y_pred)

        return acc
            

            
import numpy as np
from scipy.stats import multivariate_normal

class LinearDiscriminantAnalysis1:

    def __init__(self):
        """
        Initialize the Linear Discriminant Analysis (LDA) classifier.
        """
        self.classes = None
        self.class_priors = None
        self.class_means = None
        self.class_covs = None

    def fit(self, X, y):
        """
        Fit the LDA model to the training data.

        Parameters:
        - X: numpy.ndarray
            The training input samples.
        - y: numpy.ndarray
            The target values for the training samples.
        """
        # Get unique class labels
        self.classes = np.unique(y)

        # Compute class priors
        self.class_priors = np.array([np.mean(y == c) for c in self.classes])

        # Compute class means
        self.class_means = np.array([np.mean(X[y == c], axis=0) for c in self.classes])

        # Compute class covariance matrices
        self.class_covs = np.array([np.cov(X[y == c].T) for c in self.classes])

    def predict(self, X):
        """
        Predict class labels for input samples.

        Parameters:
        - X: numpy.ndarray
            The input samples for prediction.

        Returns:
        - numpy.ndarray
            Predicted class labels for the input samples.
        """
        discriminants = []

        # Compute discriminant function for each class
        for i, c in enumerate(self.classes):
            # Create a multivariate normal distribution for the class
            mvn = multivariate_normal(mean=self.class_means[i], cov=self.class_covs[i])
            # Compute the log likelihood of each sample for the class
            discriminant = mvn.logpdf(X) + np.log(self.class_priors[i])
            discriminants.append(discriminant)

        # Predict the class label with maximum discriminant value
        return np.argmax(discriminants, axis=0)

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the predicted labels.

        Parameters:
        - y_true: numpy.ndarray
            The true class labels.
        - y_pred: numpy.ndarray
            The predicted class labels.

        Returns:
        - float
            The accuracy of the predicted labels.
        """
        return np.mean(y_true == y_pred)


















