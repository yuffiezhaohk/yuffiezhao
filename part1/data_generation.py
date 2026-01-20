# data_generation.py
import math
import os
import itertools
from typing import List, Tuple, Dict, Any
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from choice_learn.data import ChoiceDataset

def generate_synthetic_choice_data(
    universe_size: int,
    choice_set_size: int,
    obs_per_set: int,
    filepath: str
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple, np.ndarray], List[Tuple]]:
    """
    Generates or loads synthetic discrete choice data based on a Dirichlet-Multinomial process.

    This function ensures that every possible combination of the choice set is generated exactly once
    to form a complete 'ground truth' dataset.

    Args:
        universe_size (int): Total number of items available (J).
        choice_set_size (int): Number of items in each choice set (K).
        obs_per_set (int): Number of simulated user choices per choice set.
        filepath (str): Path to the .npz file for caching results.

    Returns:
        Tuple containing:
            - availability_vectors (np.ndarray): Input features (One-hot encoding of available items).
            - selection_vectors (np.ndarray): Target labels (One-hot encoding of chosen items).
            - empirical_freqs (Dict): Mapping of ChoiceSet tuple to empirical probability distribution.
            - all_choice_sets (List): List of all unique choice set tuples.
    """

    # 1. Check Cache
    if os.path.exists(filepath):
        print(f"Loading cached data from {filepath}...")
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct complex objects from numpy arrays
        empirical_freqs = data['empirical_freqs'].item()
        # Ensure keys are tuples (immutable) for dictionary lookup
        choice_sets_list = [tuple(cs) for cs in data['choice_sets']]

        return data['X'], data['y'], empirical_freqs, choice_sets_list

    # 2. Setup Generators
    print("Cache not found. initializing data generation...")
    try:
        total_combinations = math.comb(universe_size, choice_set_size)
    except AttributeError:
        # For Python < 3.8 compatibility if needed
        import scipy.special
        total_combinations = int(scipy.special.comb(universe_size, choice_set_size))

    print(f"Generating {total_combinations} unique choice sets. This may take time.")

    item_indices = range(universe_size)
    # Create a generator for all possible combinations C(J, K)
    combination_generator = itertools.combinations(item_indices, choice_set_size)

    availability_vectors = []
    selection_vectors = []
    empirical_freqs = {}

    # Dirichlet distribution for generating ground truth probabilities
    # Concentration = 1 implies a uniform prior over the simplex
    dirichlet_dist = tfp.distributions.Dirichlet(concentration=tf.ones(choice_set_size))

    # 3. Main Generation Loop
    for current_set in tqdm(combination_generator, total=total_combinations, desc="Processing Combinations"):
        current_set_tuple = tuple(current_set)

        # Create input feature: Vector of size J where 1 indicates item is available
        avail_vec = np.zeros(universe_size, dtype='float32')
        avail_vec[list(current_set)] = 1.0

        # Sample Ground Truth probabilities for this specific set
        true_probs = dirichlet_dist.sample().numpy()

        # Simulate agent choices based on Ground Truth
        simulated_choices_indices = np.random.choice(
            choice_set_size,
            size=obs_per_set,
            p=true_probs
        )

        # Record empirical frequencies (Ground Truth for Validation RMSE)
        counts = np.bincount(simulated_choices_indices, minlength=choice_set_size)
        empirical_freqs[current_set_tuple] = counts / obs_per_set

        # Generate Training Pairs (X, y)
        for relative_index in simulated_choices_indices:
            # Map relative index (0..K-1) back to absolute item index (0..J-1)
            absolute_item_index = current_set_tuple[relative_index]

            label_vec = np.zeros(universe_size, dtype='float32')
            label_vec[absolute_item_index] = 1.0

            availability_vectors.append(avail_vec)
            selection_vectors.append(label_vec)

    # 4. Finalize and Save
    X_final = np.array(availability_vectors)
    y_final = np.array(selection_vectors)
    choice_sets_final = list(empirical_freqs.keys())

    print(f"Saving generated data to {filepath}...")
    np.savez_compressed(
        filepath,
        X=X_final,
        y=y_final,
        empirical_freqs=empirical_freqs,
        choice_sets=choice_sets_final
    )

    return X_final, y_final, empirical_freqs, choice_sets_final


def calculate_layer_width(
    network_depth: int,
    target_param_count: int,
    universe_size: int
) -> int:
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
        raise ValueError(f"Network depth must be >= 1, got {network_depth}")

    # Coefficients for the quadratic equation: ax^2 + bx + c = 0
    quad_coeff_a = network_depth - 1
    lin_coeff_b = 2 * universe_size
    const_coeff_c = -target_param_count

    # Handle linear model case (Depth=1)
    if quad_coeff_a == 0:
        return int(target_param_count / lin_coeff_b) if lin_coeff_b > 0 else 0

    discriminant = lin_coeff_b**2 - 4 * quad_coeff_a * const_coeff_c

    if discriminant < 0:
        return 0

    # Quadratic formula
    width = (-lin_coeff_b + math.sqrt(discriminant)) / (2 * quad_coeff_a)

    return int(width)
