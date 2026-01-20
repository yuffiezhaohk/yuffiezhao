# main_execution.py
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
from data_generation import generate_synthetic_choice_data, calculate_layer_width
from deephalo import build_featureless_deep_halo_model, DistributionMatchingRMSECallback

# --- Configuration ---
# Toggle this: True for quick unit testing (DEBUG), False for full paper reproduction.
DEBUG_MODE: bool = False

if DEBUG_MODE:
    print("⚠️ RUNNING IN DEBUG MODE: Small scale for verification.")
    CONFIG = {
        "universe_size": 5,         # J
        "choice_set_size": 3,       # K
        "obs_per_set": 10,
        "epochs": 5,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "param_budgets": [500, 1000],
        "network_depths": [2, 3]    # L
    }
    DATA_FILENAME = "debug_data_final.npz"
else:
    print("🚨 RUNNING IN FULL MODE: Full scale paper reproduction.")
    CONFIG = {
        "universe_size": 20,
        "choice_set_size": 15,
        "obs_per_set": 80,
        "epochs": 500,
        "learning_rate": 1e-4,
        "batch_size": 1024,
        "param_budgets": [200000, 500000],
        "network_depths": [3, 4, 5, 6, 7]
    }
    DATA_FILENAME = "full_scale_data_final.npz"


def run_comparative_experiment(
    x_ np.ndarray,
    y_ np.ndarray,
    empirical_lookup: Dict,
    choice_sets_list: List,
    config: ModelConfig
) -> Dict[int, Dict[int, Any]]:
    """
    Orchestrates the training of multiple models across different parameter budgets and depths.

    Args:
        x_data (np.ndarray): Training features.
        y_data (np.ndarray): Training labels.
        empirical_lookup (Dict): Ground truth data for validation.
        choice_sets_list (List): List of choice sets for validation.
        config (ModelConfig): Dictionary containing hyperparameters.

    Returns:
        Dict: Nested dictionary containing experiment results keyed by [Budget][Depth].
    """
    experiment_results = {}

    # Determine validation frequency
    validation_interval = 1 if DEBUG_MODE else 10

    for budget in config["param_budgets"]:
        experiment_results[budget] = {}

        for depth in config["network_depths"]:
            print(f"\n{'='*40}")
            print(f"Experiment: Budget={budget}, Depth={depth}")
            print(f"{'='*40}")

            # 1. Calculate Topology Constraints
            layer_width = calculate_layer_width(
                network_depth=depth,
                target_param_count=budget,
                universe_size=config["universe_size"]
            )

            if layer_width <= 0:
                print(f"Skipping configuration: Cannot satisfy budget {budget} with depth {depth}.")
                continue

            print(f"-> Architecture: {layer_width} neurons per layer.")

            # 2. Initialize Model
            model = build_featureless_deep_halo_model(
                universe_size=config["universe_size"],
                hidden_width=layer_width,
                network_depth=depth
            )

            # 3. Compile
            # We use MSE because we are regressing towards a probability distribution
            model.compile(
                optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
                loss='mean_squared_error'
            )

            # 4. Setup Validator
            rmse_validator = DistributionMatchingRMSECallback(
                unique_choice_sets=choice_sets_list,
                empirical_distributions=empirical_lookup,
                check_interval=validation_interval
            )

            # 5. Train
            model.fit(
                x_data, y_data,
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                callbacks=[rmse_validator],
                verbose=0  # Suppress default Keras bar to keep output clean
            )

            # 6. Log Results
            final_score = rmse_validator.rmse_history[-1] if rmse_validator.rmse_history else 0.0
            experiment_results[budget][depth] = {
                'rmse_history': rmse_validator.rmse_history,
                'final_rmse': final_score,
                'layer_width': layer_width
            }
            print(f"-> Result: Final RMSE = {final_score:.5f}")

    return experiment_results


def plot_performance_metrics_repro(results_ Dict, depths: List[int], is_debug: bool):
    """
    Plots the comparative results
    """
    # --- Setup Style ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    check_interval = 1 if is_debug else 100
    budgets = sorted(results_data.keys())
    n_depths = len(depths)

    # Define Gradient Palettes
    blues_palette = sns.color_palette("Blues", n_colors=n_depths + 2)[2:]
    oranges_palette = sns.color_palette("YlOrBr", n_colors=n_depths + 2)[2:]

    # Map Budget -> Palette
    palette_map = {
        budgets[0]: blues_palette,      # e.g. 200k or 500 (debug)
        budgets[1]: oranges_palette     # e.g. 500k or 1000 (debug)
    } if len(budgets) >= 2 else {budgets[0]: blues_palette}

    # ==========================================
    # Figure: Effect of Model Depth
    # ==========================================
    plt.figure(figsize=(8, 6))

    # Line Styles
    styles = {
        budgets[0]: {'color': '#4c72b0', 'marker': 'o', 'label': f'{int(budgets[0]/1000)}k'},
        budgets[1]: {'color': '#dd8452', 'marker': '*', 'label': f'{int(budgets[1]/1000)}k'}
    } if len(budgets) >= 2 else {budgets[0]: {'color': '#4c72b0', 'marker': 'o', 'label': str(budgets[0])}}

    for budget in budgets:
        # Extract y-values (RMSE) for x-axis (Depths)
        rmse_values = []
        valid_depths = []
        for d in depths:
            if d in results_data[budget]:
                rmse_values.append(results_data[budget][d]['final_rmse'])
                valid_depths.append(d)

        # Plot Line
        plt.plot(
            valid_depths,
            rmse_values,
            marker=styles[budget]['marker'],
            color=styles[budget]['color'],
            label=styles[budget]['label'],
            linewidth=1.5,
            markersize=6
        )

    plt.xlabel('Depth')
    plt.ylabel('Training RMSE')
    plt.xticks(depths)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================
    # Figure: Training Loss Curve Across Epochs
    # ==========================================
    plt.figure(figsize=(10, 7))

    for b_idx, budget in enumerate(budgets):
        current_palette = palette_map[budget]

        # Sort depths to ensure color gradient matches depth order
        sorted_experiment_depths = sorted(results_data[budget].keys())

        for i, depth in enumerate(sorted_experiment_depths):
            history = results_data[budget][depth]['rmse_history']
            epochs_x = np.arange(1, len(history) + 1) * check_interval

            # Assign color from palette based on depth index
            # If debug mode has fewer depths, we pick from the beginning of palette
            color = current_palette[i] if i < len(current_palette) else current_palette[-1]

            label_str = f"{int(budget/1000)}k Dep {depth}"

            plt.plot(
                epochs_x,
                history,
                color=color,
                label=label_str,
                linewidth=1.5,
                alpha=0.9
            )

    plt.xlabel('Epochs')
    plt.ylabel('Training RMSE')

    # Create a legend that fits well (top right)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, which='major', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    total_start_time = time.time()
    print(f"--- STARTING FULL PIPELINE (DEBUG_MODE={DEBUG_MODE}) ---")

    # 1. Load/Generate Data (Uses CONFIG settings from Cell 1)
    print("\n>>> STEP 1: Preparing Data...")
    X_train, y_train, empirical_freqs, all_choice_sets = generate_synthetic_choice_data(
        CONFIG["universe_size"],
        CONFIG["choice_set_size"],
        CONFIG["obs_per_set"],
        DATA_FILENAME
    )

    # 2. Run Experiments (Uses functions defined in Cells 5 & 6)
    print("\n>>> STEP 2: Running Comparative Experiments...")
    experiment_results = run_comparative_experiment(
        X_train,
        y_train,
        empirical_freqs,
        all_choice_sets,
        CONFIG
    )

    # 3. Plot Results (Uses function defined in Cell 7)
    print("\n>>> STEP 3: Visualizing Results...")
    # Pass the global DEBUG_MODE to the plotter for correct check interval calculation
    plot_performance_metrics_repro(experiment_results, CONFIG["network_depths"], DEBUG_MODE)

    print(f"\n--- PIPELINE COMPLETED in {time.time() - total_start_time:.2f} seconds ---")