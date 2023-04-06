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

def mse(y_target , y_pred):
    return np.mean(np.square(y_target - y_pred))

def mse_prime(y_target , y_pred):
    return (2 * (y_pred - y_target)) / y_target.shape[0]


def mse_prime_BGD(y_target , y_pred):
    return 2 * (y_pred - y_target)

def mse_BGD(y_target , y_pred):
    return np.square(y_target - y_pred)


def binary_cross_entropy(y_true, y_pred):
    return -(1 / np.size(y_true)) * np.sum((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

def binary_cross_entropy_prime(y_true, y_pred):
    return (1 / np.size(y_true)) * (((1-y_true) / (1- y_pred)) - (y_true / y_pred))
