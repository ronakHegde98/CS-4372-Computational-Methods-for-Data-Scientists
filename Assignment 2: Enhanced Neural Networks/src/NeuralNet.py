#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
import pandas as pd
import numpy as np
from preprocess import preprocessor

class NeuralNet:

    def __init__(self, X_train, y_train, h=4):
        #np.random.seed(1)
        # h represents the number of neurons in the hidden layers
        
        self.X = X_train
        self.y = y_train

        # Find number of input and output layers from the dataset
        input_layer_size = self.X.shape[1]

        # if not isinstance(self.y[0], np.ndarray):
        #     self.output_layer_size = 1
        # else:
        #     self.output_layer_size = 1
        
        self.output_layer_size = 1

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

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
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def __tanh_derivative(self, x):
        return 1-(np.tanh(x))**2
    
    def __relu_derivative(self,x):
        return (x>0)*1

    # Below is the training function
    def train(self, activation, max_iterations=10, learning_rate=0.25):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, activation)

            update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
            update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

            update_weight_hidden = learning_rate * np.dot(self.X.T, self.deltaHidden)
            update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b
        
            print(f"Error for iteration {iteration} is {np.sum(error)}")

        # print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        # print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        # print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        # print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        # print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))


    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden

        # TODO: I have coded the sigmoid activation, you have to do the rest
        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        elif activation == "tanh":
            self.X_hidden = self.__tanh(in_hidden)
        elif activation == "relu":
            self.X_hidden = self.__relu(in_hidden)

        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output

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

    # TODO: Implement other activation functions

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
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))
        elif activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__tanh_derivative(self.X_hidden))
        elif activation == "relu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__relu_derivative(self.X_hidden))
            
        self.deltaHidden = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, X_test, y_test):
        pass


if __name__ == "__main__":
    dataset_url = "https://raw.githubusercontent.com/ronakHegde98/CS-4372-Computational-Methods-for-Data-Scientists/master/data/diabetic_data.csv"
    df = pd.read_csv(dataset_url)
    X_train, X_test, y_train, y_test = preprocessor(df)
    y_train = y_train.values.reshape(y_train.shape[0], 1)
    y_test = y_test.values.reshape(y_test.shape[0],1)

    nn_model = NeuralNet(X_train,y_train, h=2 )
    nn_model.train(activation="relu")

    # nn_model.predict(X_test, y_test)


    # neural_network = NeuralNet("train.csv")
    # neural_network.train()
    # testError = neural_network.predict()
    # print("Test error = " + str(testError))