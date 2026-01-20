# deephalo.py
import math
import time
import itertools
import os
import unittest
import random
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def set_global_seeds(seed=42):
    """Initialize random seeds for TensorFlow, NumPy, and Python random module."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_global_seeds(42)  # Set reproducible baseline for all subsequent operations

# --- Type Aliases for Clarity ---
ChoiceSet = Tuple[int, ...]  # Represents a tuple of item indices
ModelConfig = Dict[str, Any]


class DistributionMatchingRMSECallback(Callback):
    """
    A custom Keras Callback to evaluate the Root Mean Squared Error (RMSE)
    between predicted probabilities and ground-truth empirical frequencies.

    Unlike standard validation which checks single-sample accuracy, this callback:
    1. Iterates through ALL unique choice sets.
    2. Predicts the full probability distribution for each set.
    3. Compares it against the true Dirichlet distribution from data generation.
    """

    def __init__(
        self,
        unique_choice_sets: List[ChoiceSet],
        empirical_distributions: Dict[ChoiceSet, np.ndarray],
        check_interval: int = 10
    ):
        """
        Initialize the callback.

        Args:
            unique_choice_sets (List[ChoiceSet]): List of all unique choice set tuples.
            empirical_distributions (Dict): Lookup table for ground truth probabilities.
            check_interval (int): Epoch frequency to run this expensive calculation.
        """
        super().__init__()
        self.unique_choice_sets = unique_choice_sets
        self.empirical_distributions = empirical_distributions
        self.check_interval = check_interval
        self.rmse_history: List[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        # Skip evaluation if not the correct interval
        if (epoch + 1) % self.check_interval != 0:
            return

        # Accumulator for squared errors
        batch_squared_errors = []

        # NOTE: We iterate one by one for clarity, but batched inference
        # would be used in a production environment for speed.
        for choice_set in self.unique_choice_sets:
            # Construct input vector for this specific choice set
            input_vec = np.zeros((1, self.model.input_shape[1]), dtype='float32')
            input_vec[0, list(choice_set)] = 1.0

            # Get model prediction
            pred_probs = self.model.predict(input_vec, verbose=0)[0]

            # Filter predictions to only relevant items
            relevant_preds = pred_probs[list(choice_set)]

            # Get Ground Truth
            true_freqs = self.empirical_distributions[choice_set]

            # Compute Squared Error for this set
            error = np.square(relevant_preds - true_freqs)
            batch_squared_errors.append(np.mean(error))

        # Compute global RMSE
        current_rmse = np.sqrt(np.mean(batch_squared_errors))
        self.rmse_history.append(current_rmse)

        print(f" — [Epoch {epoch+1}] Validation Distribution RMSE: {current_rmse:.6f}")


def build_featureless_deep_halo_model(
    universe_size: int,
    hidden_width: int,
    network_depth: int
) -> models.Model:
    """
    Constructs the DeepHalo neural network architecture designed for featureless discrete choice.

    Key Architectural Features:
    1.  **Quadratic Activation**: Uses x^2 instead of ReLU to explicitly model interaction effects.
    2.  **ResNet Topology**: Uses skip connections for gradient flow.
    3.  **Availability Masking**: Ensures probability of unavailable items is strictly zero.

    Args:
        universe_size (int): Size of the item universe (Input/Output dimension).
        hidden_width (int): Number of neurons in hidden layers.
        network_depth (int): Total number of non-linear transformations.

    Returns:
        models.Model: A compiled Keras functional model.
    """

    # Input: Binary vector indicating item availability
    inputs = layers.Input(shape=(universe_size,), name='Availability_Input')

    # Initial Projection: Linear mapping to hidden dimension
    # Note: use_bias=False is used to adhere strictly to parameter budget constraints logic
    x = layers.Dense(hidden_width, use_bias=False, name='Projection_Layer')(inputs)

    # Stacked Residual Blocks
    for i in range(1, network_depth + 1):
        # Identity path
        shortcut = x

        # Quadratic Activation: Explicitly models pairwise interactions
        x = layers.Lambda(lambda t: tf.square(t), name=f'Quadratic_Activation_{i}')(x)

        # Linear Transformation (Mixing)
        x = layers.Dense(hidden_width, use_bias=False, name=f'Mixing_Layer_{i}')(x)

        # Residual Connection
        x = layers.Add(name=f'Residual_Add_{i}')([shortcut, x])

    # Output Projection: Map back to item universe size
    logits = layers.Dense(universe_size, use_bias=False, name='Logit_Projection')(x)

    # --- Masking Mechanism ---
    # We must ensure the model cannot assign probability to items not in the input set.
    # 1. Cast input (0/1 floats) to boolean mask
    mask_boolean = layers.Lambda(
        lambda t: tf.cast(t, dtype=tf.bool),
        name='Create_Boolean_Mask'
    )(inputs)

    # 2. Apply Mask: Replace logits of unavailable items with -1e9 (approx negative infinity)
    # This ensures exp(logit) is effectively 0 during Softmax
    masked_logits = layers.Lambda(
        lambda args: tf.where(args[0], args[1], -1e9),
        output_shape=(universe_size,),
        name='Apply_Availability_Mask'
    )([mask_boolean, logits])

    # Probability Output
    outputs = layers.Softmax(name='Probability_Distribution')(masked_logits)

    return models.Model(inputs=inputs, outputs=outputs, name="DeepHalo_Model")