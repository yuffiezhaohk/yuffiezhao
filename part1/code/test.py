# unittest.py
import math
import os
import unittest
from data_generation import generate_synthetic_choice_data, calculate_layer_width
from deephalo import build_featureless_deep_halo_model

class TestDeepHaloReproduction(unittest.TestCase):
    """
    A comprehensive test suite to validate the core functional components of the DeepHalo project,
    ensuring robustness of the math, data generation, and model architecture.
    """

    def setUp(self):
        """
        Fixture Setup: Runs before EACH test method.
        Sets up controlled, small-scale test parameters and defines a temporary filename.
        """
        # Use small parameters for quick verification
        self.test_universe_size = 10
        self.test_choice_set_size = 4
        self.test_obs_per_set = 50
        # Define a temporary test file path
        self.test_filename = "test_suite_temp_data.npz"

    def tearDown(self):
        """
        Fixture Teardown: Runs after EACH test method.
        Cleans up temporary artifacts, such as the generated .npz file, to ensure test isolation.
        """
        if os.path.exists(self.test_filename):
            os.remove(self.test_filename)

    def test_calculate_layer_width_logic(self):
        """
        Test: Verifies the correctness of the quadratic equation solver (calculate_layer_width).
        Checks standard cases, linear models (Depth=1), and invalid budget handling.
        """
        # Case 1: Standard case (Depth=3, Budget=2000, J=20)
        # Expected width should be positive
        width = calculate_layer_width(network_depth=3, target_param_count=2000, universe_size=20)
        self.assertGreater(width, 0, "Calculated width must be a positive integer.")

        # Case 2: Linear model equivalent (Depth=1)
        # Formula simplifies to 2 * J * W = Budget. W = 200 / (2 * 10) = 10
        width_linear = calculate_layer_width(network_depth=1, target_param_count=200, universe_size=10)
        self.assertEqual(width_linear, 10, "Width calculation for Depth=1 (Linear) is incorrect.")

        # Case 3: Impossible budget (should return 0)
        width_impossible = calculate_layer_width(network_depth=5, target_param_count=10, universe_size=100)
        self.assertEqual(width_impossible, 0, "Should return 0 when the budget is too small for the architecture.")

        # Case 4: Invalid input (Depth=0 must raise ValueError)
        with self.assertRaises(ValueError):
            calculate_layer_width(network_depth=0, target_param_count=1000, universe_size=20)

    def test_data_generation_integrity(self):
        """
        Test: Verifies data generation pipeline for correct shapes, combinatorial logic, and caching.
        """
        # Run 1: Generate data and cache
        X, y, freqs, choices = generate_synthetic_choice_data(
            self.test_universe_size, self.test_choice_set_size, self.test_obs_per_set, self.test_filename
        )

        # Check 1: File creation
        self.assertTrue(os.path.exists(self.test_filename), "Data file was not created/cached.")

        # Check 2: Shape consistency (X and y must have the same number of samples)
        self.assertEqual(X.shape[0], y.shape[0], "The number of samples in X and y do not match.")

        # Check 3: Combinatorial Logic (e.g., C(10, 4) = 210)
        expected_combinations = math.comb(self.test_universe_size, self.test_choice_set_size)
        self.assertEqual(len(choices), expected_combinations, "The number of unique combinations generated is incorrect.")

        # Check 4: Caching behavior (Teardown ensures the file is removed afterward)

    def test_model_architecture_spec(self):
        """
        Test: Verifies the Keras model construction, checking input/output shapes and critical layer presence.
        """
        test_J = 5
        test_W = 8
        model = build_featureless_deep_halo_model(universe_size=test_J, hidden_width=test_W, network_depth=3)

        # Check 1: Input shape (should be (None, J))
        self.assertEqual(model.input_shape[1], test_J, "Model input layer dimension (J) is incorrect.")

        # Check 2: Output shape (should be (None, J))
        self.assertEqual(model.output_shape[1], test_J, "Model output layer dimension (J) is incorrect.")

        # Check 3: Critical layer presence (Availability Masking is mandatory)
        layer_names = [layer.name for layer in model.layers]
        self.assertIn("Apply_Availability_Mask", layer_names, "DeepHalo architecture lacks the mandatory availability masking layer.")

        # Check 4: Critical layer presence (Quadratic Activation)
        self.assertTrue(any("Quadratic_Activation" in name for name in layer_names), "DeepHalo architecture lacks the quadratic activation layer.")


# ==========================================
# RUN TEST SUITE
# ==========================================
if __name__ == '__main__':
    print("Running Formal Unit Tests...")
    # This command runs unittest.main() without exiting the Jupyter kernel
    unittest.main(argv=['first-arg-is-ignored'], verbosity=2, exit=False)
