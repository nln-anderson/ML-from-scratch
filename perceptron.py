# Functions and stuff for ML

import numpy as np
from enum import Enum

# Activation Functions:

def sigmoid(x: float) -> float:
    """Sigmoid activiation function

    Args:
        x (float): Input to the function.

    Returns:
        float: Output of sigmoid function.
    """
    return (1/(1+np.exp(-x)))

def relu(x: float) -> float:
    """ReLU activation function

    Args:
        x (float): value to be plugged in

    Returns:
        float: output of function
    """
    return max(0,x)

class Activation(enumerate):
    sigmoid = 1
    ReLU = 2

# Classes:
class Perceptron:
    """ Perceptron/neuron class. Building block of neural networks
    """
    # Instance vars
    num_weights: int # Number of weights
    weights: np.ndarray # Array/vector of weights for perceptron
    activation: Activation

    def __init__(self, num_weights: int, alpha = 0.1, activation = sigmoid) -> None:
        self.activation = activation
        self.alpha = alpha
        self.num_weights = num_weights
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.weights = np.random.randn(self.num_weights+1) # Plus 1 accounts for bias

    def step_function(self, x: float) -> int:
        if x>0:
            return 1
        else:
            return 0

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int) -> None:
        """Trains the perceptron on the data. Updates weights and bias accordingly

        Args:
            X (np.ndarray): Input data
            Y (np.ndarray): Labels for the input data
            epochs (int): Number of training iterations to perform
        """
        bias_term = np.ones((X.shape[0], 1))  # Create a column vector of ones for bias
        X = np.hstack((X, bias_term))  # Add the bias term to input data

        # Now the training
        for epoch in range(epochs):
            for row in range(X.shape[0]):
                wx = np.dot(self.weights, X[row])
                prediction = self.step_function(wx)
                error = prediction - Y[row]

                if error != 0:
                    self.weights += -self.alpha * error * X[row]

    def predict(self, X: np.ndarray) -> int:
        """Predicts an output given data

        Args:
            X (np.ndarray): An entry of data to test the model.

        Returns:
            int: The prediction made by the model.
        """
        # Add the bias term to input data
        X = np.append(X, [1])
        return self.step_function(np.dot(X, self.weights))
    
"Testing Perceptron--------------------------------------"
# OR function
perc = Perceptron(2)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [1]])

test = np.array([[1,1]])
perc.fit(X,Y, epochs=1000)

print(perc.weights)
print(perc.predict(test))

# OR with more inputs
perc = Perceptron(5)

X = np.array([[0,0,0,0,0], [1,0,1,1,0], [1,1,1,1,1], [0,1,1,0,0], [0,0,0,0,1]])
Y = np.array([[0], [1], [1], [1], [1]])

perc.fit(X,Y, epochs=1000)

test = np.array([0,0,0,0,0])
print(perc.predict(test))
