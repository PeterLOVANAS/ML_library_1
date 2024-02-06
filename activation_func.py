from activation import Activation
import numpy as np
from layer import Layer

# This is for all activations function
class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - (np.tanh(x)**2)
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: np.exp(x) / np.power((np.exp(x) + 1) , 2)
        super().__init__(sigmoid, sigmoid_prime)



class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)
        super().__init__(relu , relu_prime)



"""
Softmax activation

f(x) = e^x_i / ∑ x_{j = 1}^n e^x_j
This equation means that for a single output neuron of the softmax activation layer is depended on all values of input neurons X
The importance is that the number of neurons in X layer and Y layer is always the same as other activation layers.
Lastly, the softmax activation layer usually used as the final output layer in multi-class classification problem that want to get the output which is in the range of [0,1].


We know that ∂E/∂x_k = ∑ ∂E/∂y_i * ∂y_i/∂x_k
    1) ∂E/∂y_i  has no problem as that could later be generalized as ∂E/∂Y
    2) ∂y_i/∂x_k has two possible answers depend on according conditions;
            2.1) IF k = i ; ∂y_i/∂x_k  = y_i *(1 - y_i) 
            2.2) IF k != i ; ∂y_i/∂x_k = -y_i * y_k   <= Meaning: Depends on each X neuron 

As we generalized, we came to that 
∂E/∂X = (M * (I-M.T)) @ ∂E/∂Y
Note:
M is the matrix of [[y_i, y_i, y_i, ..., y_i] , ... ,[y_n , y_n, y_n, ... ,y_n]] Which has shape of (n,n). Pay attention that y is the output of the model (output of self.forward())
I is the identity matrix (1 along the diagonal) which has shape of (n,n)
∂E/∂Y is the output gradient (Compute using loss function) which has shape of (n,1)


"""



class Softmax(Layer):

    def forward(self, input):
        exp_vec = np.exp(input) # vector of e^x_i
        self.output = exp_vec / np.sum(exp_vec)
        return self.output

    def backward(self, output_gradient):
        k_n = self.output.shape[0]
        M = np.tile(self.output , k_n)  # Intuition: b = np.array([[1], [2], [3]]) => np.tile(b , 3) => output: array([[1 ,1 ,1] , [2 ,2 ,2] , [3, 3 ,3]])
        return (M * (np.identity(k_n) - M.T)) @ output_gradient  # return ∂E/∂X to the previous layer

