import numpy as np
import warnings

def normalize_image(array, epsilon=1e-8):
    """
    Normalizes a numpy array (image) along the height and width axes using the formula: output = (array - mean) / (std + epsilon).

    Parameters:
    array (numpy.ndarray): Input array (image) to be normalized.
    epsilon (float): A small value added to the denominator to avoid division by zero. Default is 1e-8.

    Returns:
    numpy.ndarray: Normalized array (image).
    """
    mean = array.mean(axis=(0, 1))  # Calculate mean along height and width axes
    std = array.std(axis=(0, 1))  # Calculate standard deviation along height and width axes

    # Check for division by zero
    zero_std_channels = np.where(std == 0)
    if zero_std_channels[0].size > 0:
        std[zero_std_channels] = epsilon

    # Normalize the array
    output = (array - mean) / std  # Normalize the array using the calculated mean and standard deviation

    return output
