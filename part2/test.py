# unittest.py
import unittest
import numpy as np
import tensorflow as tf
from data_generation import SimulationExperiment
from blp import BLPEstimator
from bayesian_mcmc import BayesianShrinkageEstimator

class TestBLPPipeline(unittest.TestCase):
    def test_simulation_shape(self):
        T, N, J = 5, 100, 3
        sim = SimulationExperiment(T, N, J, dgp_type=1, seed=42)
        dataset = sim.run()
        feats = dataset.items_features_by_choice[0]
        self.assertEqual(feats.shape, (T * N, J + 1, 2))

    def test_blp_estimator_run(self):
        T, N, J = 5, 100, 3
        sim = SimulationExperiment(T, N, J, dgp_type=1, seed=42)
        dataset = sim.run()
        blp = BLPEstimator(dataset, T, J, 'no_cost_iv')
        val = blp.compute_gmm_objective(1.5)
        self.assertIsInstance(val, float)

    def test_bayesian_mcmc_step(self):
        T, N, J = 5, 100, 3
        sim = SimulationExperiment(T, N, J, dgp_type=1, seed=42)
        dataset = sim.run()
        # FIXED: Correct Class Name
        mcmc = BayesianShrinkageEstimator(dataset, T, J)
        p, eta, gamma = mcmc.fit(n_iter=10, burn_in=5)
        self.assertEqual(len(p), 4)
        self.assertEqual(eta.shape, (T, J))
        self.assertEqual(gamma.shape, (T, J))

if __name__ == '__main__':
    print("Running Unit Tests...")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)