# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import VALIDATION_INTERVAL

def plot_experiment_results(results, config):
    """
    Generates the two main figures:
    1. Performance vs. Network Depth
    2. Training RMSE evolution over Epochs
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    budgets = sorted(results.keys())
    depths = config["network_depths"]

    # --- Figure 1: Effect of Depth ---
    plt.figure(figsize=(8, 6))
    colors = ['#4c72b0', '#dd8452']
    markers = ['o', '*']

    for i, budget in enumerate(budgets):
        rmse_values = [results[budget][d]['final'] for d in depths if d in results[budget]]
        plt.plot(depths, rmse_values, marker=markers[i], color=colors[i], 
                 label=f'Budget: {int(budget/1000)}k', linewidth=1.5)

    plt.xlabel('Network Depth (L)')
    plt.ylabel('Final Validation RMSE')
    plt.title('Effect of Model Depth on Choice Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Figure 2: Training Convergence ---
    plt.figure(figsize=(10, 7))
    for i, budget in enumerate(budgets):
        # Create a gradient palette for depths within this budget
        palette = sns.color_palette("Blues" if i == 0 else "Oranges", n_colors=len(depths) + 2)[2:]
        
        for j, depth in enumerate(sorted(results[budget].keys())):
            history = results[budget][depth]['history']
            epochs_x = np.arange(1, len(history) + 1) * VALIDATION_INTERVAL
            
            plt.plot(epochs_x, history, color=palette[j], 
                     label=f"{int(budget/1000)}k | Depth {depth}", alpha=0.8)

    plt.xlabel('Epochs')
    plt.ylabel('Validation RMSE')
    plt.title('RMSE Convergence Across Architectures')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True, which='major', alpha=0.4)
    plt.tight_layout()
    plt.show()