import os
import math
import itertools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from choice_learn.data import ChoiceDataset

def get_choice_dataset(universe_size, choice_set_size, obs_per_set, filepath):
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
    if os.path.exists(filepath):
        data = np.load(filepath, allow_pickle=True)
        return ChoiceDataset(
            shared_features_by_choice=data['shared'],
            items_features_by_choice=data['items'],
            available_items_by_choice=data['avail'],
            choices=data['choices']
        ), data['empirical_freqs'].item(), [tuple(cs) for cs in data['choice_sets']]

    # Setup generation
    total_combos = math.comb(universe_size, choice_set_size)
    item_indices = range(universe_size)
    combo_gen = itertools.combinations(item_indices, choice_set_size)
    
    all_avail, all_choices, empirical_freqs = [], [], {}
    dirichlet = tfp.distributions.Dirichlet(concentration=tf.ones(choice_set_size))

    for current_set in tqdm(combo_gen, total=total_combos, desc="Generating Data"):
        current_set_tuple = tuple(current_set)
        avail_vec = np.zeros(universe_size)
        avail_vec[list(current_set)] = 1.0
        
        true_probs = dirichlet.sample().numpy()
        empirical_freqs[current_set_tuple] = true_probs
        
        sim_indices = np.random.choice(choice_set_size, size=obs_per_set, p=true_probs)
        for rel_idx in sim_indices:
            all_avail.append(avail_vec)
            all_choices.append(current_set_tuple[rel_idx])

    # Choice-learn expects (N_choices, N_items, N_features). 
    # Since we are featureless, we provide a dummy feature dimension of 1.
    items_features = np.ones((len(all_choices), universe_size, 1))
    shared_features = np.zeros((len(all_choices), 1)) # Dummy shared
    
    choices_indices = np.array(all_choices)
    avail_matrix = np.array(all_avail)

    # Save and Return
    np.savez_compressed(filepath, shared=shared_features, items=items_features, 
                        avail=avail_matrix, choices=choices_indices, 
                        empirical_freqs=empirical_freqs, choice_sets=list(empirical_freqs.keys()))

    ds = ChoiceDataset(shared_features_by_choice=shared_features, 
                       shared_features_by_choice_names=["dummy_shared"],
                       items_features_by_choice=items_features, 
                       items_features_by_choice_names=["intercept"],
                       available_items_by_choice=avail_matrix, 
                       choices=choices_indices)
    return ds, empirical_freqs, list(empirical_freqs.keys())