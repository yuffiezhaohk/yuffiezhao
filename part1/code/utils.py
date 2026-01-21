import math
import random
import os
import numpy as np
import tensorflow as tf

def set_global_seeds(seed=42):
    """Initialize random seeds for TensorFlow, NumPy, and Python random module."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def calculate_layer_width(network_depth, target_param_count, universe_size):
    """
    Calculates the optimal number of neurons per hidden layer to satisfy a total parameter constraint.

    This function solves the quadratic equation derived from the ResNet architecture parameter count:
        (Depth - 1) * width^2 + 2 * universe_size * width - target_params = 0

    Args:
        network_depth (int): The number of layers in the network (L). Must be >= 1.
        target_param_count (int): The maximum allowable number of parameters (Budget).
        universe_size (int): The total number of unique items in the discrete choice set (J).

    Returns:
        int: The calculated width (number of neurons) for the hidden layers. Returns 0 if no valid solution exists.

    Raises:
        ValueError: If network_depth is less than 1.
    """
    if network_depth < 1:
        raise ValueError("Network depth must be >= 1")
    quad_coeff_a = network_depth - 1
    lin_coeff_b = 2 * universe_size
    const_coeff_c = -target_param_count

    if quad_coeff_a == 0:
        return int(target_param_count / lin_coeff_b) if lin_coeff_b > 0 else 0
    discriminant = lin_coeff_b**2 - 4 * quad_coeff_a * const_coeff_c
    if discriminant < 0: return 0
    width = (-lin_coeff_b + math.sqrt(discriminant)) / (2 * quad_coeff_a)
    return int(width)