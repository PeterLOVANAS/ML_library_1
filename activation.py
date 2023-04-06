from layer import Layer
import numpy as np


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

Backpropagation [ 3) ]
Define X = Y  ( Y means output layer 2). ) as input
Define Y = f(X) (Y means output of this activation later and f as an activation function)
As  ∂E/∂X = ∂E/∂Y ⊙ f'(X) 

"""

# This is base class for activation layer
class Activation(Layer):
    def __init__(self, activation_function , activation_prime):
        super().__init__()
        self.activation = activation_function
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input  # Input of the activation layer which is Y (according the above theory)
        return self.activation(self.input) # The output of this layer then become F = f(Y)

    def backward(self, output_gradient , learning_rate):
        return output_gradient * self.activation_prime(self.input)  # self.input is X
