import time
from config import CONFIG, DATA_FILENAME, SEED
from utils import set_global_seeds
from data_loader import get_choice_dataset
from experiment import run_experiment
from visualization import plot_experiment_results
from export import save_experiment_outputs

def main():
    # 0. Initialize
    set_global_seeds(SEED)
    start_time = time.time()
    
    print(f"{'='*50}")
    print("      DEEPHALO REPRODUCTION PIPELINE")
    print(f"{'='*50}")

    # 1. Data Formulation with choice-learn
    print("\n[STEP 1] Formulating Dataset...")
    dataset, empirical_dist, choice_sets = get_choice_dataset(
        CONFIG["universe_size"], 
        CONFIG["choice_set_size"], 
        CONFIG["obs_per_set"], 
        DATA_FILENAME
    )

    # 2. Training Orchestration
    print("\n[STEP 2] Training Models (Comparative Study)...")
    results = run_experiment(dataset, empirical_dist, choice_sets, CONFIG)

    # 3. Exporting Metadata
    print("\n[STEP 3] Exporting Results...")
    export_folder = save_experiment_outputs(results, CONFIG)

    # 4. Visualization
    print("\n[STEP 4] Generating Performance Plots...")
    plot_experiment_results(results, CONFIG)

    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETED SUCCESSFULY")
    print(f"Total Duration: {total_time/60:.2f} minutes")
    print(f"Artifacts saved in: {export_folder}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()