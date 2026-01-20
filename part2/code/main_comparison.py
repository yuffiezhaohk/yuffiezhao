import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize_scalar
import tqdm as tqdm_module  # Rename module to avoid conflict
import gc
import logging
from data_generation import SimulationExperiment
from blp import BLPEstimator
from bayesian_mcmc import BayesianShrinkageEstimator

# Configure Environment
tf.keras.backend.set_floatx('float64')
logging.getLogger().setLevel(logging.ERROR)

class MetricsCalculator:
    TRUE_PARAMS = np.array([-1.0, -1.0, 0.5, 1.5])

    @staticmethod
    def compute_row(results, dgp_type, method_type):
        res_matrix = np.array(results)
        param_bias = np.mean(res_matrix[:, :4] - MetricsCalculator.TRUE_PARAMS, axis=0)
        param_sd = np.std(res_matrix[:, :4], axis=0)
        xi_bias = np.mean(res_matrix[:, 4])
        xi_sd = np.std(res_matrix[:, 4])
        row_bias = list(param_bias) + [xi_bias]
        row_sd = list(param_sd) + [xi_sd]
        if method_type == 'shrink' and dgp_type in [1, 2]:
            row_bias.append(np.mean(res_matrix[:, 5]))
            row_sd.append(np.mean(res_matrix[:, 6]))
        else:
            row_bias.append(np.nan)
            row_sd.append(np.nan)
        return row_bias, row_sd

def run_full_factorial_replication():
    """Main execution loop for the full factorial experiment."""

    # Experiment Configuration
    T_list = [25, 100]
    J_list = [5, 15]
    dgp_list = [1, 2, 3, 4]
    n_reps = 50

    total_scenarios = len(T_list) * len(J_list) * len(dgp_list)
    print(f"Starting Full Replication: {total_scenarios} scenarios, {n_reps} reps each.")

    for T in T_list:
        for J in J_list:
            print(f"\n{'='*60}")
            print(f"SCENARIO: Markets (T)={T}, Products (J)={J}")
            print(f"{'='*60}")

            for dgp in dgp_list:
                print(f"\n>>> Running DGP {dgp} ...")

                # Containers for results
                results = {'blp_cost': [], 'blp_nocost': [], 'shrink': []}

                # FIXED: Explicitly call tqdm.tqdm to resolve TypeError
                iterator = tqdm_module.tqdm(range(n_reps), leave=False, desc=f"Sim T{T}J{J}D{dgp}")

                for r in iterator:
                    # 1. Generate Data
                    # Unique seed for every combination
                    seed = (T * 100000) + (J * 1000) + (dgp * 100) + r
                    sim = SimulationExperiment(T, 1000, J, dgp, seed)
                    dataset = sim.run()
                    true_eta_dev = dataset.true_eta.reshape(-1)

                    # 2. BLP (Cost IV)
                    blp1 = BLPEstimator(dataset, T, J, 'cost_iv')
                    opt1 = minimize_scalar(blp1.compute_gmm_objective, bounds=(0.1, 4.0), method='bounded')
                    xi_b1 = np.mean(np.abs(blp1.last_xi - true_eta_dev))
                    results['blp_cost'].append([*blp1.last_beta, opt1.x, xi_b1])

                    # 3. BLP (No Cost IV)
                    blp2 = BLPEstimator(dataset, T, J, 'no_cost_iv')
                    opt2 = minimize_scalar(blp2.compute_gmm_objective, bounds=(0.1, 4.0), method='bounded')
                    xi_b2 = np.mean(np.abs(blp2.last_xi - true_eta_dev))
                    results['blp_nocost'].append([*blp2.last_beta, opt2.x, xi_b2])

                    # 4. Bayesian Shrinkage
                    mcmc = BayesianShrinkageEstimator(dataset, T, J)
                    est_p, est_eta_tot, est_gamma = mcmc.fit(n_iter=5000, burn_in=2500)

                    est_eta_dev = est_eta_tot - np.mean(est_eta_tot)
                    xi_b3 = np.mean(np.abs(est_eta_dev - dataset.true_eta))

                    # Prob Calculation logic
                    res_row = [*est_p, xi_b3]
                    if dgp in [1, 2]:
                        flat_g = est_gamma.flatten()
                        is_nz = np.abs(true_eta_dev) > 0.1
                        p1 = np.mean(flat_g[is_nz]) if is_nz.sum() > 0 else 0
                        p2 = np.mean(flat_g[~is_nz]) if (~is_nz).sum() > 0 else 0
                        res_row.extend([p1, p2])
                    else:
                        res_row.extend([np.nan, np.nan])

                    results['shrink'].append(res_row)

                    # Cleanup
                    del dataset, sim, blp1, blp2, mcmc
                    gc.collect()

                # --- Aggregate and Display Results ---
                rows = []
                index_names = []

                # Process BLP Cost
                b, s = MetricsCalculator.compute_row(results['blp_cost'], dgp, 'blp')
                rows.extend([b, s])
                index_names.extend(['BLP(Cost)-Bias', 'BLP(Cost)-SD'])

                # Process BLP No Cost
                b, s = MetricsCalculator.compute_row(results['blp_nocost'], dgp, 'blp')
                rows.extend([b, s])
                index_names.extend(['BLP(NoCost)-Bias', 'BLP(NoCost)-SD'])

                # Process Shrinkage
                b, s = MetricsCalculator.compute_row(results['shrink'], dgp, 'shrink')
                rows.extend([b, s])
                index_names.extend(['Shrinkage-Bias', 'Shrinkage-SD'])

                df = pd.DataFrame(rows, columns=['Int', 'Bp', 'Bw', 'Sig', 'Xi', 'Prob'], index=index_names)
                print(f"Results for T={T}, J={J}, DGP={dgp}:")
                print(df.round(3).fillna('-'))

if __name__ == '__main__':
    print("\nRunning Main Experiment...")
    run_full_factorial_replication()
