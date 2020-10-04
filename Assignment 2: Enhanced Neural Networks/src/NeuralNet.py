from copy import deepcopy
import numpy as np
import pandas as pd
import argparse
from preprocess import preprocessor

class NeuralNet:

    def __init__(self, X_train, y_train, h=4):
        #np.random.seed(1)
        # h represents the number of neurons in the hidden layers
        self.X = X_train
        self.y = y_train

        # Find number of input and output layers from the dataset
        input_layer_size = self.X.shape[0]
        
        
        self.output_layer_size = 1

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((h, input_layer_size)) - 1
        self.Wb_hidden = 2 * np.random.random((h,1)) - 1

        self.W_output = 2 * np.random.random((self.output_layer_size,h)) - 1
        self.Wb_output = np.ones((self.output_layer_size,1))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h
            

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self,x)
        elif activation == "relu":
            self.__relu(self,x)
     

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self,x)
        elif activation == "relu":
            self.__relu_derivative(self,x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def __tanh(self, x):
        return np.tanh(x)
    
    def __relu(self, x):
        return np.maximum(0, x)

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self, x):
        return 1-(np.tanh(x))**2
    
    def __relu_derivative(self,x):
        return (x>0)*1


    # Below is the training function
    def train(self, activation, max_iterations=300, learning_rate=0.00001, momentum = 0.90):
        
        update_weight_output, update_weight_output_b, update_weight_hidden, update_weight_hidden_b = 0,0,0,0
        
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            
            error = 0.5 * np.power((out - self.y), 2)
            
            
            self.past_delta = [deepcopy(update_weight_output),
                                        deepcopy(update_weight_output_b),
                                        deepcopy(update_weight_hidden),
                                        deepcopy(update_weight_hidden_b)]
            
            
            self.backward_pass(out, activation)
            
            update_weight_output = learning_rate * (1-momentum) * np.dot(self.deltaOut,self.X_hidden.T) + momentum*self.past_delta[0]
            
            update_weight_output_b = learning_rate * (1-momentum) * np.dot(self.deltaOut, np.ones((np.size(self.X, 1), 1))) + momentum*self.past_delta[1]
            
            update_weight_hidden = learning_rate * (1-momentum)* np.dot(self.deltaHidden,self.X.T) + momentum*self.past_delta[2]
            
            update_weight_hidden_b = learning_rate * (1-momentum)* np.dot(self.deltaHidden,np.ones((np.size(self.X, 1), 1))) + momentum*self.past_delta[3]

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b
            

        print("Training: After " + str(max_iterations) + " iterations, the total error is " + str(np.average(np.sum(error))))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.W_hidden, self.X) + self.Wb_hidden

        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        elif activation == "tanh":
            self.X_hidden = self.__tanh(in_hidden)
        elif activation == "relu":
            self.X_hidden = self.__relu(in_hidden)

        in_output = np.dot(self.W_output, self.X_hidden) + self.Wb_output
        
        # output 
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        elif activation == "tanh":
            out = self.__tanh(in_output)
        elif activation == "relu":
            out = self.__relu(in_output)
        return out


    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)
        

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation):
        
        if activation == "sigmoid":
            delta_hidden_layer = (self.W_output.T.dot(self.deltaOut)) * (self.__sigmoid_derivative(self.X_hidden))
        elif activation == "tanh":
            delta_hidden_layer = (self.W_output.T.dot(self.deltaOut)) * (self.__tanh_derivative(self.X_hidden))
        elif activation == "relu":
            delta_hidden_layer = (self.W_output.T.dot(self.deltaOut)) * (self.__relu_derivative(self.X_hidden))
        
        self.deltaHidden = delta_hidden_layer


    def predict(self, X_test, y_test, activation):
        predict_hidden = np.dot(self.W_hidden, X_test) + self.Wb_hidden
        
        self.X_hidden = self.__relu(predict_hidden)
        
        if(activation == "sigmoid"):
            self.X_hidden = self.__sigmoid(predict_hidden)
        elif(activation=="relu"):
            self.X_hidden = self.__relu(predict_hidden)
        elif(activation == "tanh"):
            self.X_hidden = self.__tanh(predict_hidden)
        
        predict_output = np.dot(self.W_output, self.X_hidden) + self.Wb_output
        
        if(activation == "sigmoid"):
            out = self.__sigmoid(predict_output)
        elif(activation=="relu"):
            out = self.__relu(predict_output)
        elif(activation == "tanh"):
            out = self.__tanh(predict_output)
        
        
        error = 0.5 * np.power((out - y_test), 2)
        print(f"Error on Test Dataset is {np.sum(error)}")
        return out
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Neural Network')
    parser.add_argument("--hidden", type = int, help = "number of hidden neurons",
    default = 5)
    parser.add_argument("--activation", help = "activation for neural network", default= "sigmoid")
   
    args = parser.parse_args()
    h = int(args.hidden) 
    activation = args.activation

    dataset_url = "https://raw.githubusercontent.com/ronakHegde98/CS-4372-Computational-Methods-for-Data-Scientists/master/data/diabetic_data.csv"
    df = pd.read_csv(dataset_url)
    X_train, X_test, y_train, y_test = preprocessor(df)

    #reshaping of train and test
    y_train = y_train.values.reshape(y_train.shape[0], 1)
    y_test = y_test.values.reshape(y_test.shape[0],1)

    nn_model = NeuralNet(X_train.T,y_train.T, h)
    nn_model.train(activation)

    predictions = nn_model.predict(X_test.T,y_test.T,activation)
    # predictions = np.around(predictions, 0).astype(np.int32)

