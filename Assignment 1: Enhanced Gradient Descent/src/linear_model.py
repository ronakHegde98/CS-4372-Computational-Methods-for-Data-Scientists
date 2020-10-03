import pandas as pd
import numpy as np

class LinearRegression:
    def __init__(self, iterations = 1000, learning_rate = 0.01, beta = 0.9, momentum=True):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.momentum = momentum
        self.costs = np.zeros(iterations)
        self.beta = beta
        
    def fit(self, X_train, y_train):
        num_examples, num_features = X_train.shape
        self.weights = np.ones(num_features) 
        vdW = np.zeros(num_features)
        
        for iteration in range(self.iterations):
            weighted_sum = np.dot(X_train, self.weights)
            loss = weighted_sum - y_train
            cost = np.sum(np.square(loss))/(2*num_examples)
            
            #Gradient w.r.t to parameters
            dW = np.dot(X_train.T, loss)/num_examples
            
            #momentum update
            if(self.momentum == True):
                vdW = (self.beta*vdW) + (1-self.beta)*dW
                self.weights -= self.learning_rate*vdW
            else:
                self.weights -= self.learning_rate*dW

            self.costs[iteration] = cost
            
    def predict(self,X_test):
        return np.dot(X_test, self.weights)    
    
    def mean_squared_error(self, predictions, true_values):
        return np.sum(np.square(predictions-true_values))/len(true_values)

    def all_metrics(self, predictions, true_values):
        print('Mean Squared Error', self.mean_squared_error(predictions, true_values))
        print(f'model weights are {self.weights}')