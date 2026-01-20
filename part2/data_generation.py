import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from choice_learn.data import ChoiceDataset

tfd = tfp.distributions

class SimulationExperiment:
    """
    Simulates market data for BLP experiments including Endogeneity and Sparsity.
    """
    def __init__(self, n_markets: int, n_consumers: int, n_items: int,
                 dgp_type: int, seed: int):
        self.T = n_markets
        self.N = n_consumers
        self.J_real = n_items
        self.J_total = n_items + 1  # Index 0 is the outside option
        self.dgp_type = dgp_type

        self.true_params = {
            'beta_p': -1.0,
            'beta_w': 0.5,
            'sigma': 1.5,
            'xi_bar': -1.0
        }

        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _generate_structural_error(self):
        if self.dgp_type in [1, 2]:  # Sparse settings
            n_nonzero = int(0.4 * self.J_real)
            eta_vals = [1.0 if (j + 1) % 2 != 0 else -1.0 for j in range(n_nonzero)]
            eta_vals += [0.0] * (self.J_real - n_nonzero)
            eta = tf.tile(tf.constant([eta_vals], dtype=tf.float64), [self.T, 1])
        else:  # Non-sparse settings
            eta = tf.random.normal((self.T, self.J_real), 0.0, 1.0 / 3.0, dtype=tf.float64)
        return eta

    def _calculate_price_endogeneity(self, eta):
        alpha = tf.zeros_like(eta)
        if self.dgp_type in [1, 3]:  # Exogenous
            return alpha

        if self.dgp_type == 2:  # Sparse Endogenous
            alpha = tf.where(tf.abs(eta - 1.0) < 1e-5, 0.3, alpha)
            alpha = tf.where(tf.abs(eta + 1.0) < 1e-5, -0.3, alpha)
        elif self.dgp_type == 4:  # Non-Sparse Endogenous
            alpha = tf.where(eta >= (1.0 / 3.0), 0.3, alpha)
            alpha = tf.where(eta <= (-1.0 / 3.0), -0.3, alpha)
        return alpha

    def run(self):
        w_real = tf.cast(tfp.distributions.Uniform(1.0, 2.0).sample((self.T, self.J_real)), tf.float64)
        eta_real = self._generate_structural_error()
        xi_real = self.true_params['xi_bar'] + eta_real
        u_shock = tf.random.normal((self.T, self.J_real), 0.0, 0.7, dtype=tf.float64)
        alpha = self._calculate_price_endogeneity(eta_real)
        p_real = alpha + 0.3 * w_real + u_shock

        zeros_col = tf.zeros((self.T, 1), dtype=tf.float64)
        p_all = tf.concat([zeros_col, p_real], axis=1)
        w_all = tf.concat([zeros_col, w_real], axis=1)
        xi_all = tf.concat([zeros_col, xi_real], axis=1)

        beta_pi = tf.random.normal((self.T, self.N, 1), self.true_params['beta_p'], self.true_params['sigma'], dtype=tf.float64)
        p_exp = tf.expand_dims(p_all, 1)
        w_exp = tf.expand_dims(w_all, 1)
        xi_exp = tf.expand_dims(xi_all, 1)

        V_ijt = (beta_pi * p_exp) + (self.true_params['beta_w'] * w_exp) + xi_exp
        epsilon = tf.cast(tfp.distributions.Gumbel(0.0, 1.0).sample((self.T, self.N, self.J_total)), tf.float64)
        choices = tf.argmax(V_ijt + epsilon, axis=2, output_type=tf.int32)

        return self._package_dataset(p_exp, w_exp, choices, u_shock, eta_real)

    def _package_dataset(self, p_exp, w_exp, choices, u_shock, eta_real):
        total_obs = self.T * self.N
        choices_flat = tf.reshape(choices, [-1]).numpy()
        p_flat = tf.reshape(tf.tile(p_exp, [1, self.N, 1]), [total_obs, self.J_total, 1])
        w_flat = tf.reshape(tf.tile(w_exp, [1, self.N, 1]), [total_obs, self.J_total, 1])
        items_features = tf.concat([p_flat, w_flat], axis=2).numpy()

        dataset = ChoiceDataset(
            items_features_by_choice=items_features,
            choices=choices_flat,
            available_items_by_choice=np.ones((total_obs, self.J_total), int),
            items_features_by_choice_names=["price", "weight"]
        )
        dataset.true_u_shock = u_shock.numpy()
        dataset.true_eta = eta_real.numpy()
        return dataset
