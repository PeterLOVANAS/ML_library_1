from layer import Layer
import numpy as np
from utility import clip_gradient_by_norm



"""
Theory

Splitting a single dense layer into 
1)  Input layer => X
2)  Output layer => Y
3)  Activation layer => f(Y)  
These are sub_layer of the ANN. By the way, we'll just call it as layer.

Definition
Define Y = [ y_1 ,y_2, y_3 ,..., y_j]  <= Shape: (j,1)
Define X = [ x_1 ,x_2, x_3 ,..., x_i]  <= Shape: (i,1) 
Define W <= Shape: (j,i)
    Define w_i_j  as the element in W  (i is index of input neurons , j is index of output neurons)
Define B = [ b_1 ,b_2, b_3 ,..., b_j]  <= Shape: (j,1)
then   Y = (W @ X) + B

Principle
1)  X of L_m = Y of L_n (while L_m or L_n is the layer of indices m and n , Input of this layer is the output of the previous one)
2)  ∂E/∂Y_n = ∂E/∂X_m  (This is also true for derivatives)


Backpropagation [ 1) and 2) ]
Suppose we're given ∂E/∂Y {Output layer}
∂E/∂Y = [ ∂E/∂y_1 , ∂E/∂y_2 , ∂E/∂y_3 ,... ,∂E/∂y_j ]  <= Shape: (j,1)
then we need to find
1)  ∂E/∂W <= Shape: (j,i)  
2)  ∂E/∂B <= Shape: (j,1)
3)  ∂E/∂X <= Shape: (i,1) {Input layer}

We're trying to find each by write them in term of ∂E/∂Y (The reason is that ∂E/∂Y will be the input of our backward())

Eq1 (Finding ∂E/∂W) 
As ∂E/∂w_j_i = (∂E/∂y_j)* x_i  (Input(i)-Output (j) pairs of neuron)
then   ∂E/∂W = ∂E/∂Y @ X.T    <= M.T  means transpose of matrix, M

Eq2 (Finding ∂E/∂B)
As ∂E/∂b_j = ∂E/∂y_j  (Only bias of that output neuron matter)
then   ∂E/∂B = ∂E/∂Y

Eq3 (Finding ∂E/∂X) 
As ∂E/∂x_i = ∑ (∂E/∂y_j)* w_j_i   <= This summation is run by j due to index, i is constant as we're finding derivative with respect to specific x_i
then   ∂E/∂X = W.T @ ∂E/∂Y

"""


class Dense(Layer):
    def __init__(self, input_size, output_size):  # Number of neurons in input or output of the layer
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size , 1)

    def forward(self, input):  # weights: W , input: X , output: Y
        self.input = input
        return (self.weights @ self.input) + self.bias  # Y = (W @ X) + B  => Output vector

    def backward(self, output_gradient , learning_rate):  # output_gradient is ∂E/∂Y
        weights_gradient = output_gradient @ self.input.T   # Eq1: weights_gradient = ∂E/∂W  (j,i)
        input_gradient = self.weights.T @ output_gradient # ∂E/∂X which is the output gradient of a previous layer of the ANN
        weights_gradient = clip_gradient_by_norm(weights_gradient , 1e+50)
        self.weights -= learning_rate * weights_gradient  # learning_rate is a scalar and W and ∂E/∂W has the same shape
        self.bias -= learning_rate * output_gradient  # Eq2
        return input_gradient
