import unittest
import os
import math
import numpy as np
from choice_learn.data import ChoiceDataset

# Import from our new modules
from config import CONFIG
from utils import calculate_layer_width, set_global_seeds
from data_loader import get_choice_dataset
from model import build_deep_halo

class TestDeepHaloModular(unittest.TestCase):
    """
    Validation suite for the modular DeepHalo implementation.
    """

    def setUp(self):
        """
        Fixture Setup: Uses small parameters for rapid testing.
        """
        set_global_seeds(42)
        self.test_universe = 6
        self.test_set_size = 3
        self.test_obs = 10
        self.test_filename = "unit_test_temp.npz"

    def tearDown(self):
        """
        Fixture Teardown: Clean up generated files.
        """
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_math_solver(self):
        """
        Test: Verifies the quadratic width solver for parameter budgeting.
        """
        # Linear case: Budget 200, Universe 10, Depth 1 -> 2 * 10 * W = 200 -> W = 10
        width = calculate_layer_width(1, 200, 10)
        self.assertEqual(width, 10)
        
        # Impossible case: Budget too small for deep network
        width_small = calculate_layer_width(10, 5, 100)
        self.assertEqual(width_small, 0)

    def test_choice_learn_integration(self):
        """
        Test: Ensures get_choice_dataset returns a valid ChoiceDataset object.
        """
        dataset, empirical, sets = get_choice_dataset(
            self.test_universe, self.test_set_size, self.test_obs, self.test_filename
        )

        # Type Check
        self.assertIsInstance(dataset, ChoiceDataset)
        
        # Combinatorial Integrity: C(6, 3) = 20
        expected_combos = math.comb(self.test_universe, self.test_set_size)
        self.assertEqual(len(sets), expected_combos)
        
        # Shape Check: N_obs = combos * obs_per_set
        expected_rows = expected_combos * self.test_obs
        self.assertEqual(len(dataset), expected_rows)
        
        # Check ChoiceDataset internal consistency
        # Choice-learn choices should be indices within [0, universe_size - 1]
        self.assertTrue(np.max(dataset.choices) < self.test_universe)

    def test_model_masking(self):
        """
        Test: Verifies the DeepHalo masking logic.
        """
        test_width = 8
        model = build_deep_halo(self.test_universe, test_width, 2)
        
        # Create a sample input where only items 0, 1, and 2 are available
        sample_input = np.zeros((1, self.test_universe))
        sample_input[0, [0, 1, 2]] = 1.0
        
        prediction = model.predict(sample_input, verbose=0)[0]
        
        # Items 3, 4, 5 MUST have 0 probability due to masking
        for i in range(3, self.test_universe):
            self.assertAlmostEqual(prediction[i], 0.0, places=7)
            
        # The sum of probabilities for available items must be 1.0
        self.assertAlmostEqual(np.sum(prediction), 1.0, places=5)

if __name__ == '__main__':
    print("Running Modular System Tests...")
    unittest.main()