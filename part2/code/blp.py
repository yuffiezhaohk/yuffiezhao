import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from choice_learn.data import ChoiceDataset

tfd = tfp.distributions

class BLPEstimator:
    """
    Implements the Berry-Levinsohn-Pakes (BLP) estimator using GMM.

    Attributes:
        iv_strategy (str): 'cost_iv' (strong) or 'no_cost_iv' (weak).
    """

    def __init__(self, dataset: ChoiceDataset, T: int, J_real: int, iv_strategy: str):
        self.T = T
        self.J_real = J_real
        self.J_total = J_real + 1

        self.p, self.w = self._extract_features(dataset)
        self.s_obs, self.s_outside = self._compute_shares(dataset)
        self.Z, self.X = self._construct_instruments_and_regressors(dataset, iv_strategy)

        # Pre-compute GMM weight matrix parts
        self.inv_ZZ = np.linalg.pinv(self.Z.T @ self.Z)

        # Fixed random draws for integration
        self.nu_i = tf.random.normal((1, 500, 1), dtype=tf.float64)

        # Storage for results
        self.last_beta = None
        self.last_xi = None

    def _extract_features(self, dataset):
        """Extracts price and weight matrices from ChoiceDataset."""
        feat_array = dataset.items_features_by_choice[0]
        raw_feats = np.array(feat_array).reshape(self.T, -1, self.J_total, 2)
        # Market features are constant across consumers, take 0th index
        # Slice [:, 1:, :] to exclude outside option (index 0)
        p = raw_feats[:, 0, 1:, 0].astype(np.float64)
        w = raw_feats[:, 0, 1:, 1].astype(np.float64)
        return p, w

    def _compute_shares(self, dataset):
        """Computes observed market shares from consumer choices."""
        choices_mat = dataset.choices.reshape(self.T, -1)
        counts = np.array([np.bincount(choices_mat[t], minlength=self.J_total) for t in range(self.T)])
        total_N = counts.sum(axis=1, keepdims=True)

        s_outside = np.maximum(counts[:, 0:1] / total_N, 1e-8)
        s_inside = np.maximum(counts[:, 1:] / total_N, 1e-8)
        return s_inside, s_outside

    def _construct_instruments_and_regressors(self, dataset, iv_strategy):
        """Constructs Regressor matrix X and Instrument matrix Z."""
        ones = np.ones((self.T, self.J_real))
        # X = [1, p, w]
        X = np.stack([ones, self.p, self.w], axis=2).reshape(-1, 3)

        # Z construction
        if iv_strategy == 'cost_iv':
            u = dataset.true_u_shock
            Z_stack = [ones, self.w, self.w**2, u, u**2]
        else:
            Z_stack = [ones, self.w, self.w**2, self.w**3, self.w**4]

        Z = np.stack(Z_stack, axis=2).reshape(-1, 5)
        return Z, X

    def solve_delta(self, sigma: float) -> np.ndarray:
        """
        Solves for mean utility (delta) using Contraction Mapping.
        """
        # Berry Inversion initialization
        delta = tf.convert_to_tensor(np.log(self.s_obs) - np.log(self.s_outside), dtype=tf.float64)
        p_tf = tf.constant(self.p, dtype=tf.float64)

        for _ in range(2000):
            # mu = sigma * nu * p
            mu = sigma * self.nu_i * tf.expand_dims(p_tf, 1)

            exp_val = tf.exp(tf.expand_dims(delta, 1) + mu)
            denom = 1.0 + tf.reduce_sum(exp_val, axis=2, keepdims=True)
            s_pred = tf.reduce_mean(exp_val / denom, axis=1)

            # Update step
            delta_new = delta + np.log(self.s_obs) - tf.math.log(s_pred + 1e-9)

            if tf.reduce_max(tf.abs(delta_new - delta)) < 1e-12:
                break
            delta = delta_new

        return delta.numpy()

    def compute_gmm_objective(self, sigma: float) -> float:
        """
        Calculates the GMM objective function value for a given sigma.
        Concentrates out linear parameters (beta) via 2SLS.
        """
        if sigma <= 0.01 or sigma > 5.0:
            return 1e10

        delta = self.solve_delta(sigma).reshape(-1)

        # 2SLS Projection: X_hat = P_Z * X
        # beta = (X_hat' X)^-1 X_hat' delta
        Pz = self.Z @ self.inv_ZZ @ self.Z.T
        X_hat = Pz @ self.X

        try:
            beta = np.linalg.solve(X_hat.T @ self.X, X_hat.T @ delta)
        except np.linalg.LinAlgError:
            return 1e10

        # Structural error
        xi = delta - self.X @ beta

        # GMM Objective: xi' Z (Z'Z)^-1 Z' xi
        val = (self.Z.T @ xi).T @ self.inv_ZZ @ (self.Z.T @ xi)

        # Store for retrieval
        self.last_beta = beta
        self.last_xi = xi
        return val
