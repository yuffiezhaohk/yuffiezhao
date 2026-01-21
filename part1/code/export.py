import os
import json
import datetime

def save_experiment_outputs(results, config, base_dir="exports"):
    """
    Saves trained model weights (if passed) and experiment metadata to a local directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join(base_dir, f"experiment_{timestamp}")
    
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Save Configuration
    with open(os.path.join(export_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Save Results Summary (Stripping the long history for a clean JSON)
    summary = {}
    for budget, depths in results.items():
        summary[str(budget)] = {
            str(d): {"final_rmse": float(info["final"])}  # 👈 关键修复：转为 float
            for d, info in depths.items()
        }

    with open(os.path.join(export_path, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f">>> Exported experiment metadata to: {export_path}")
    return export_path