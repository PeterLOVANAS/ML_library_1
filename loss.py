import numpy as np

# This is for all of loss functions
"""
Theory

1) MSE (Mean square error)
    Define y*_i  to be elements of Y*  (The target vector of each sample in dataset,D) 
    Define y_i   to be elements of Y   (The predicted vector of the model for each sample)
    
    E = (1/n)*[∑(y*_i  - y_i )**2]   (eq.1)
    while n is the number of samples (first element in the shape tuple of y*_i)
    
    For the last layer, it need to calculate its ∂E/∂Y
    Based on what we know in eq.1, we could calculate the derivative for each element like this
    ∂E/∂y_i = (2/n) * (y_i - y*_i) 
    
    then ∂E/∂Y = (2/n) * (Y - Y*)

2) Binary cross entropy
    Define y*_i  to be elements of Y*  (The target vector of each sample in dataset,D) 
    Define y_i   to be elements of Y   (The predicted vector of the model for each sample)
    
    E = -(1/n) * ∑ y*_i * log(y_i)  + (1-y*_i) * log(1-y_i) 
    while n is the number of samples (first element in the shape tuple of y*_i)
    
    For the last layer, it need to calculate its ∂E/∂Y
    Based on what we know in eq.1, we could calculate the derivative for each element like this
    ∂E/∂y_i = (1/n) * [((1-y*_i) / (1- y_i)) - (y*_i / y_i)] 
    
    then ∂E/∂Y (1/n) * [((1-Y*) / (1- Y)) - (Y* / Y)] 
    
"""



class loss_func:
    def __init__(self, y_target , y_pred):
        self.y_target = y_target
        self.y_pred = y_pred

    def loss(self):
        pass

    def loss_prime(self):
        pass


class mse(loss_func):
    def __init__(self, y_target , y_pred):
        super().__init__(y_target , y_pred)

    def loss(self):
        return np.mean(np.square(self.y_target - self.y_pred))

    def loss_prime(self):
        return (2 * (self.y_pred - self.y_target)) / self.y_target.shape[0]




class binary_cross_entropy(loss_func):
    def __init__(self, y_target , y_pred):
        super().__init__(y_target, y_pred)

    def loss(self):
        return -(1 / np.size(self.y_target)) * np.sum((self.y_target * np.log(self.y_pred)) + ((1 - self.y_target) * np.log(1 - self.y_pred)))

    def loss_prime(self):
        return (1 / np.size(self.y_target)) * (((1-self.y_target) / (1- self.y_pred)) - (self.y_target / self.y_pred))

