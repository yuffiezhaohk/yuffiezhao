import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm
from choice_learn.data import ChoiceDataset

tfd = tfp.distributions

class BayesianShrinkageEstimator:
    """
    Implements the Bayesian Shrinkage estimator using MCMC (Gibbs + MH).
    Uses a Spike-and-Slab prior to enforce sparsity on structural errors.
    """

    def __init__(self, dataset: ChoiceDataset, T: int, J_real: int):
        self.T = T
        self.J_real = J_real
        self.J_total = J_real + 1

        self.p, self.w, self.counts = self._extract_data(dataset)
        # X for Bayesian model: [p, w] (Intercept handled by xi_bar)
        self.X_cov = np.stack([self.p, self.w], axis=2)

        # Priors (from paper)
        self.tau0_sq = 1e-3  # Spike variance
        self.tau1_sq = 1.0   # Slab variance
        self.a_phi = 1.0
        self.b_phi = 1.0

        self.nu_i = np.random.normal(0, 1, (200, 1))

    def _extract_data(self, dataset):
        feat_array = dataset.items_features_by_choice[0]
        raw_feats = np.array(feat_array).reshape(self.T, -1, self.J_total, 2)
        p = raw_feats[:, 0, 1:, 0].astype(np.float64)
        w = raw_feats[:, 0, 1:, 1].astype(np.float64)

        choices_mat = dataset.choices.reshape(self.T, -1)
        counts = np.array([np.bincount(choices_mat[t], minlength=self.J_total) for t in range(self.T)])
        return p, w, counts

    def _compute_choice_probs(self, beta, sigma, xi_bar, eta):
        """Calculates market shares using the Logit integral."""
        # Mean utility
        xb = np.dot(self.X_cov, beta)
        delta = xb + xi_bar[:, np.newaxis] + eta

        # Random coefficients
        p = self.X_cov[..., 0]
        mu = sigma * self.nu_i[np.newaxis, :, :] * p[:, np.newaxis, :]

        # Probabilities
        exp_u = np.exp(delta[:, np.newaxis, :] + mu)
        sum_exp = 1.0 + np.sum(exp_u, axis=2, keepdims=True)
        probs = np.mean(exp_u / sum_exp, axis=1)

        # Outside option share
        s0 = 1.0 - np.sum(probs, axis=1, keepdims=True)
        return np.column_stack([s0, probs])

    def _log_likelihood(self, probs):
        return np.sum(self.counts * np.log(np.maximum(probs, 1e-10)))

    def fit(self, n_iter=2000, burn_in=1000):
        """
        Executes the MCMC sampling loop (Cold Start).
        """
        # Initialization (Random Cold Start)
        beta = np.array([-1.0, 0.5])
        sigma = 1.5
        xi_bar = np.full(self.T, -1.0)
        phi = np.full(self.T, 0.5)

        # Initialize eta randomly to avoid "zero-trap" deadlock
        eta = np.random.normal(0, 1.0, (self.T, self.J_real))
        gamma = (np.abs(eta) > 0.05).astype(float)

        curr_probs = self._compute_choice_probs(beta, sigma, xi_bar, eta)
        curr_ll = self._log_likelihood(curr_probs)

        store_params = []
        store_eta_total = []
        store_gamma = []

        # MCMC Loop
        for it in range(n_iter):
            # 1. Update Beta (Slope)
            beta, curr_ll, curr_probs = self._mh_step_beta(beta, sigma, xi_bar, eta, curr_ll, curr_probs)

            # 2. Update Sigma (Random Coefficient)
            sigma, curr_ll, curr_probs = self._mh_step_sigma(beta, sigma, xi_bar, eta, curr_ll, curr_probs)

            # 3. Update Xi_bar (Market Intercepts)
            xi_bar, curr_ll, curr_probs = self._mh_step_xi_bar(beta, sigma, xi_bar, eta, curr_ll, curr_probs)

            # 4. Update Eta (Structural Error)
            eta, curr_ll, curr_probs = self._mh_step_eta(beta, sigma, xi_bar, eta, gamma, curr_ll, curr_probs)

            # 5. Update Gamma (Spike Indicator)
            gamma = self._gibbs_step_gamma(eta, phi)

            # 6. Update Phi (Sparsity Probability)
            phi = self._gibbs_step_phi(gamma)

            # Storage with thinning
            if it >= burn_in and it % 2 == 0:
                total_xi = xi_bar[:, None] + eta
                est_int = np.mean(total_xi)
                store_params.append([est_int, beta[0], beta[1], sigma])
                store_eta_total.append(total_xi)
                store_gamma.append(gamma)

        return np.mean(store_params, 0), np.mean(store_eta_total, 0), np.mean(store_gamma, 0)

    # --- MCMC Helper Methods ---

    def _mh_step_beta(self, beta, sigma, xi_bar, eta, curr_ll, curr_probs):
        prop_beta = beta + np.random.normal(0, 0.02, 2)
        prop_probs = self._compute_choice_probs(prop_beta, sigma, xi_bar, eta)
        prop_ll = self._log_likelihood(prop_probs)

        prior_diff = (-0.5 * np.sum(prop_beta**2) / 10.0) - (-0.5 * np.sum(beta**2) / 10.0)

        if np.log(np.random.rand()) < (prop_ll - curr_ll + prior_diff):
            return prop_beta, prop_ll, prop_probs
        return beta, curr_ll, curr_probs

    def _mh_step_sigma(self, beta, sigma, xi_bar, eta, curr_ll, curr_probs):
        prop_sigma = sigma + np.random.normal(0, 0.05)
        if prop_sigma <= 0.05:
            return sigma, curr_ll, curr_probs

        prop_probs = self._compute_choice_probs(beta, prop_sigma, xi_bar, eta)
        prop_ll = self._log_likelihood(prop_probs)

        def log_prior(s): return -np.log(s) - (np.log(s)**2)

        if np.log(np.random.rand()) < (prop_ll - curr_ll + log_prior(prop_sigma) - log_prior(sigma)):
            return prop_sigma, prop_ll, prop_probs
        return sigma, curr_ll, curr_probs

    def _mh_step_xi_bar(self, beta, sigma, xi_bar, eta, curr_ll, curr_probs):
        prop_xi_bar = xi_bar + np.random.normal(0, 0.05, self.T)
        prop_probs = self._compute_choice_probs(beta, sigma, prop_xi_bar, eta)
        prop_ll = self._log_likelihood(prop_probs)

        prior_diff = (-0.5 * np.sum(prop_xi_bar**2) / 10.0) - (-0.5 * np.sum(xi_bar**2) / 10.0)

        if np.log(np.random.rand()) < (prop_ll - curr_ll + prior_diff):
            return prop_xi_bar, prop_ll, prop_probs
        return xi_bar, curr_ll, curr_probs

    def _mh_step_eta(self, beta, sigma, xi_bar, eta, gamma, curr_ll, curr_probs):
        prop_eta = eta + np.random.normal(0, 0.1, (self.T, self.J_real))
        var = np.where(gamma == 1, self.tau1_sq, self.tau0_sq)

        prior_curr = -0.5 * np.sum(eta**2 / var)
        prior_prop = -0.5 * np.sum(prop_eta**2 / var)

        prop_probs = self._compute_choice_probs(beta, sigma, xi_bar, prop_eta)
        prop_ll = self._log_likelihood(prop_probs)

        if np.log(np.random.rand()) < (prop_ll - curr_ll + prior_prop - prior_curr):
            return prop_eta, prop_ll, prop_probs
        return eta, curr_ll, curr_probs

    def _gibbs_step_gamma(self, eta, phi):
        d1 = norm.pdf(eta, 0, np.sqrt(self.tau1_sq)) * phi[:, None]
        d0 = norm.pdf(eta, 0, np.sqrt(self.tau0_sq)) * (1 - phi[:, None])
        prob_1 = d1 / (d1 + d0 + 1e-12)
        return (np.random.rand(*eta.shape) < prob_1).astype(float)

    def _gibbs_step_phi(self, gamma):
        sum_g = gamma.sum(axis=1)
        return np.random.beta(self.a_phi + sum_g, self.b_phi + (self.J_real - sum_g))
