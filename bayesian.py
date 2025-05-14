from scipy.stats import invwishart, multivariate_normal
import numpy as np

# This optimize operates under the assumption that the prior and likelihood distributions are of the same 
class ConjugatePriorOptimizer():
    def __init__(self, num_observations, mean_returns, covariance_returns, mean_prop, covariance_prop):
        # Hyperparams for raw Normal-inverse-Wishart

        # What our current estimates are based on all data
        self.current_num_observations = num_observations
        self.current_mean_returns = mean_returns
        self.current_covariance_returns = covariance_returns
    
        # From the last balance (portfolio is based on this)
        self.previous_num_observations = num_observations
        self.previous_mean_returns = mean_returns
        self.previous_covariance_returns = covariance_returns

        # Hyperparams for proportional Normal-inverse-Wsihart

        # Our current estimates
        self.current_mean_prop = mean_prop
        self.current_covariance_prop = covariance_prop

        # Our previous estimates
        self.previous_mean_prop = mean_prop
        self.previous_covariance_prop = covariance_prop

    def _calculate_proportional_mean(self, mean):
        return mean / np.sum(mean)


    # This is impossible and the approximation is probably not that good. We can test later
    # def _calculate_proportioanal_covariance(self, mean, cov):


    # We construct the prior distribution for the returns/risk with an Inverse-Wishart distribution
    def update_prior(self, sample_observation):
        self.current_covariance_returns += (sample_observation - self.current_mean_returns)@(sample_observation - self.current_mean_returns).T * (self.current_num_observations/(self.current_num_observations+1))
        self.current_mean_returns = (self.current_num_observations*self.current_mean_returns + sample_observation) / (self.current_num_observations + 1)

        sample_prop = self._calculate_proportional_mean(sample_observation)
        self.current_covariance_prop += (sample_prop - self.current_mean_prop)@(sample_prop - self.current_mean_prop).T * (self.current_num_observations/(self.current_num_observations+1))
        self.current_mean_prop = (self.current_num_observations * self.current_mean_prop + sample_prop) / (self.current_num_observations + 1)

        self.current_num_observations += 1


    # Function to take samples of our Normal-Inverse-Wishart distributions
    def _sample_niw(self, mean, cov, n_obs, num_samples):
        samples = []
        for _ in range(num_samples):
            sigma = invwishart.rvs(df=n_obs, scale=cov)
            mu = multivariate_normal.rvs(mean=mean, cov=sigma/n_obs)
            samples.append((mu, sigma))
        return samples


    def _kl_divergence(self, p_mu, p_sigma, q_mu, q_sigma, reg = False, penalty = 0.0):
        """KL divergence D_KL(P || Q) for two multivariate normals"""
        d = p_mu.shape[0]
        inv_q_sigma = np.linalg.inv(q_sigma)
        diff = q_mu - p_mu

        term1 = np.trace(inv_q_sigma @ p_sigma)
        term2 = diff.T @ inv_q_sigma @ diff
        term3 = np.log2(np.linalg.det(q_sigma) / np.linalg.det(p_sigma))
        
        if reg:
            return 0.5 * (term1 + term2 + term3 - d) + penalty * (np.linalg.norm(diff) ** 2)
        else:
            return 0.5 * (term1 + term2 + term3 - d)
    
    # Employ JS-Divergence to determine the divergence between the current 
    def _compute_js_divergence(self, prev_mean, prev_cov, curr_mean, curr_cov, reg = False, penalty = 0.0):
        # DO MONTE CARLO TO ESIMATE JS-DIVERGENCE 
        num_samples = 10000
        
        # Using the same sample from mixture distribution for comparison between new and prev
        prev_dist_samples = self._sample_niw(prev_mean, prev_cov, self.previous_num_observations, num_samples)
        curr_dist_samples = self._sample_niw(curr_mean, curr_cov, self.current_num_observations, num_samples)
        mixture_dist_samples = []

        for _ in range(num_samples):
            if np.random.rand() < 0.5:
                mixture_dist_samples.append(self._sample_niw(prev_mean, prev_cov, self.previous_num_observations, 1)[0])
            else:
                mixture_dist_samples.append(self._sample_niw(curr_mean, curr_cov, self.current_num_observations, 1)[0])

        jsd_samples = []
        for i in range(len(mixture_dist_samples)):
            mu_p, cov_p = prev_dist_samples[i]
            mu_c, cov_c = curr_dist_samples[i]
            mu_m, cov_m = mixture_dist_samples[i]

            kl_pm = self._kl_divergence(mu_p, cov_p, mu_m, cov_m, reg, penalty)
            kl_cm = self._kl_divergence(mu_c, cov_c, mu_m, cov_m, reg, penalty)

            jsd_samples.append(0.5*(kl_pm+kl_cm))

        return np.mean(jsd_samples)

    def compute_total_js_divergence(self, prop_coeff = 1.0, returns_coeff = 1.0):
        # Calculate JS Divergence for proportions
        prop_js = self._compute_js_divergence(
            self.previous_mean_prop, self.previous_covariance_prop,
            self.current_mean_prop, self.current_covariance_prop
        )

        # Calculate JS Divergence for raw prices
        returns_js = self._compute_js_divergence(
            self.previous_mean_returns, self.previous_covariance_returns,
            self.current_mean_returns, self.current_covariance_returns, True, 0.2
        )

        return prop_js, returns_js, prop_coeff * prop_js + returns_coeff * returns_js

    # RETURN TO THIS TO FIND GOOD HYPERPARAMETERS
    def compute_normalized_js_divergence(self):
        return 1/(1 + np.exp(-1 * self.compute_total_js_divergence()))

if __name__ == '__main__':

    mean_returns = np.array([5.0, 10, 20])  # Mean vector (e.g., expected returns)
    cov_returns = np.array([                  # Covariance matrix (must be symmetric and PD)
        [0.05, 0.01, 0.003],
        [0.01, 0.06, 0.005],
        [0.003, 0.005, 0.07]
    ])
    mean_prop = np.array([5/35, 10/35, 20/35])
    cov_prop = np.array([
        [ 3.00e-05, -8.00e-06, -2.20e-05],
        [-8.00e-06,  2.80e-05, -2.00e-05],
        [-2.20e-05, -2.00e-05,  4.20e-05]
    ])

    opt = ConjugatePriorOptimizer(20, mean_returns, cov_returns, mean_prop, cov_prop)
    opt.update_prior(np.array([4, 10, 20]))
    print(opt.compute_total_js_divergence()) 