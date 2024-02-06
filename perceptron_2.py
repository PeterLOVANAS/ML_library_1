import numpy as np
import matplotlib.pyplot as plt
from dense import Dense
from activation_func import Tanh , Sigmoid
from loss import mse, mse_prime , mse_prime_BGD , mse_BGD
from layer import Layer

class Perceptron(Layer):
    def __init__(self, input_size,  output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size , 1)

    def forward(self, input):
        self.input = input
        return Sigmoid().forward((self.weights @ self.input) + self.bias)

    def backward(self, error , learning_rate):
        self.weights += learning_rate * error[0][0] * self.input.reshape((1,2))
        self.bias += learning_rate * error


# Generate random input data
np.random.seed(42)
num_samples = 200
X = np.random.randn(num_samples, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)


# hyperparameters

epochs = 10
learning_rate = 0.1
perceptron = Perceptron(2 ,1)

for epoch in range(epochs):
    for i in range(num_samples):
        prediction = perceptron.forward(X[i])
        error = y[i] - prediction
        perceptron.backward(error ,learning_rate)

# Plot the data and decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('X1')
plt.ylabel('X2')

x_line = np.linspace(-3, 3, 100)

y_line = -(perceptron.weights[: , 0][0] * x_line + perceptron.bias) / perceptron.weights[: ,1][0]

plt.plot(x_line, y_line.reshape(100,), color='red')

plt.show()
