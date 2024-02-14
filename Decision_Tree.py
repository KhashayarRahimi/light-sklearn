import numpy as np
import re

class DecisionTree:

    def __init__(self,x,y):

        self.Boxes = {}
        self.counter = 1
        self.x = x
        self.y = y
        self.task = None

    def RSS(self, y_true, y_pred):

        Sum = 0

        for i in range(len(y_true)):

            Sum +=  (y_true[i] - y_pred[i])**2
        
        return Sum

    def splitter(self,x,y):

        # collect information about each feature and the RSS it achieve
        col_info = {}
        #col_S_Value = {}

        for col in range(x.shape[1]):

            # The most simple value for 'S' to divide the feature values is its mean
            #col_mean = np.mean(x[:,col])
            S_error = {}
            for s in np.unique(x[:,col]):

                col_pred = {'>=s': np.mean([y[i] for i in range(y.shape[0]) if x[i,col] >= s ]),
                            '<s': np.mean([y[i] for i in range(y.shape[0]) if x[i,col] < s ])
                            }

                y_pred = []

                for i in range(x.shape[0]):

                    if x[i,col] >= s:

                        y_pred.append(col_pred['>=s'])
                    
                    else:

                        y_pred.append(col_pred['<s'])
            
                col_RSS = self.RSS(y, y_pred)
                S_error[col_RSS] = s

            col_info[min(list(S_error.keys()))] = [col, S_error[min(list(S_error.keys()))]]
            #col_S_Value
        
        selected_col_Svalue = col_info[min(list(col_info.keys()))]

        return selected_col_Svalue
    
    # TreeMaker: Recursive Binary Splitting
    def fit(self,x,y,threshold = 5):

        if len(y) <= threshold:
 
                return self.Boxes
        else:
                
            Selected_col = self.splitter(x,y)
            #print('Selected_col',Selected_col)
            #Selected_col_mean = np.mean(x[:,Selected_col])

            BiggerSIndex = [i for i in range(x.shape[0]) if x[i,Selected_col[0]] >= Selected_col[1]]
            x1, y1 =  x[BiggerSIndex], y[BiggerSIndex]
            next_bigger_selected_col = self.splitter(x1,y1)

            SmallerSIndex = [i for i in range(x.shape[0]) if x[i,Selected_col[0]] < Selected_col[1]]
            x2, y2 =  x[SmallerSIndex], y[SmallerSIndex]
            next_smaller_selected_col = self.splitter(x2,y2)

            self.Boxes[f'{self.counter}_{Selected_col[0]}'] = Selected_col #[Selected_col,Selected_col[]]
            self.Boxes[f'{self.counter}_{Selected_col[0]}'].append({'bigger':next_bigger_selected_col[0],
                                                                'smaller':next_smaller_selected_col[0]})
            #self.Boxes[f'{self.counter}{Selected_col}'].append(np.mean(self.y))

            #if len(y) <= threshold:
            #    self.Boxes[f'{self.counter}_{Selected_col}'].append(np.mean(self.y))
            
            self.counter = self.counter + 1
            #print('bigger',f'{self.counter}_{next_bigger_selected_col}')
            self.Boxes[f'{self.counter}_{next_bigger_selected_col[0]}'] = [next_bigger_selected_col[1]]#[np.mean(x1[:,next_bigger_selected_col])]
            if len(y1) <= threshold:
                #print(f'{self.counter}_{next_bigger_selected_col}','len(y1) <= threshold')
                self.Boxes[f'{self.counter}_{next_bigger_selected_col[0]}'].append(np.mean(y1))



            #print('smaller',f'{self.counter}_{next_smaller_selected_col}',[np.mean(x2[:,next_smaller_selected_col])])
            self.Boxes[f'{self.counter}_{next_smaller_selected_col[0]}'] = [next_smaller_selected_col[1]]#[np.mean(x2[:,next_smaller_selected_col])]
            if len(y2) <= threshold:
                #print(f'{self.counter}_{next_smaller_selected_col}','len(y2) <= threshold')
                self.Boxes[f'{self.counter}_{next_smaller_selected_col[0]}'].append(np.mean(y2))
            
            self.fit(x1, y1, threshold=threshold)
            self.fit(x2, y2, threshold=threshold)

            return  self.Boxes
        
    def predict(self,Box, x_test):
        #Box = self.fit(self.x, self.y)

        prediction = []
        Node = list(Box.keys())[0]

        for sample_indx in range(x_test.shape[0]):
            sample = x_test[sample_indx]
            print('sample:',sample)
            Node = list(Box.keys())[0]

            while type(Box[Node][-1]) == dict:
                selected_col = Box[Node][0]
                print(Box[Node][-1],selected_col)

                if sample[selected_col] >= Box[Node][1]:
                    Node = str(int(re.match(r"^(\d+)_", Node).group(1)) + 1) +'_' + str(Box[Node][2]["bigger"])
                else:
                    Node = str(int(re.match(r"^(\d+)_", Node).group(1)) + 1) +'_' + str(Box[Node][2]["smaller"])

            print('Node:',Node, Box[Node][-1])
            prediction.append(Box[Node][-1])

        return prediction
    
    
    def accuracy(self, y_true, y_pred):

        if self.task == 'classification':

            return np.mean(y_true == y_pred)
        
        else:
            MAE = 0
            for sample in range(y_true.shape[0]):

                MAE += abs(y_true[sample] - y_pred[sample])
            
            MAE = MAE/(y_true.shape[0])

            return MAE

            
#---------------------------------------
        
"""
Example:

import Decision_Tree
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
        
        
X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)

# Add more samples
additional_samples = 500
X_additional, y_additional = make_regression(n_samples=additional_samples, n_features=8, noise=0.1, random_state=42)

# Concatenate additional samples to existing dataset
X = np.vstack([X, X_additional])
y = np.concatenate([y, y_additional])

# Split the dataset into training and testing sets
test_size = 0.3  # 30% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
#DT = Decision_Tree.DecisionTree(X_train,y_train)
DT = DecisionTree(X_train,y_train)

Box = DT.fit(X_train,y_train,threshold=5)

y_pred = DT.predict(Box,X_test)

print('acc:',DT.accuracy(y_test,y_pred))

"""