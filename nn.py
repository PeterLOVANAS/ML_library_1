import numpy as np
from convolution import Convolution

class Module:
    def __init__(self):
        self.layers = None # []  stack from first layer to the last
        self.input = None
        self.output = None

    def forward(self, input):
        pass


    def parameters(self):
        parameters = []
        for l in self.layers:
            param_arr = np.array(l.parameters())
            parameters.append(param_arr)

        return np.array(parameters , dtype= object)


    def backward(self , loss_grad):  # Backpropagation through entire model after getting loss gradient
        dw_gradient = []
        db_gradient = []
        all_layer_grad = [] # both weights and biases of the entire model
        output_grad_layer = loss_grad
        for l in reversed(self.layers):
            
            output_grad_layer = l.backward(output_grad_layer)
            if l.get_gradients() == None:  # Activation function
                pass

            else:
                dw_gradient.append(l.get_gradients()[0])
                db_gradient.append(l.get_gradients()[1])

        all_layer_grad.append(reversed(dw_gradient))
        all_layer_grad.append(reversed(db_gradient))

        return all_layer_grad # Return the gradients of the entire model






"""
for x_batch , y_batch in dataset:
    avg_grad = []
    for x , y in x_batch , y_batch:
        
        output = x
        for l in layers:
            output = l.forward(output)
        
        grad = loss_prime(output , y)
        for l in reversed(layers):
            grad = l.backward(grad)
        
        avg_grad.append()
        
    avg_grad = sum(avg_grad) / len(x_batch)
     
    for l in layers:
        grad_l = np.array(l.get_gradients)
        param_l = l.parameters()
        param_l += optimizer(grad_l)  # Not yet!
"""























