from scipy.stats import invwishart, multivariate_normal
import numpy as np

# This optimize operates under the assumption that the prior and likelihood distributions are of the same 
class ConjugatePriorOptimizer():
    def __init__(self, num_observations, mean_returns, covariance_returns):
        # Hyperparams for Normal-inverse-Wishart

        # What our current estimates are based on all data
        self.current_num_observations = num_observations
        self.current_mean_returns = mean_returns
        self.current_covariance_returns = covariance_returns
    
        # From the last balance (portfolio is based on this)
        self.previous_num_observations = num_observations
        self.previous_mean_returns = mean_returns
        self.previous_covariance_returns = covariance_returns


    # We construct the prior distribution for the returns/risk with an Inverse-Wishart distribution
    def update_prior(self, sample_observation):
        self.current_covariance_returns += (sample_observation - self.current_mean_returns)@(sample_observation - self.current_mean_returns).T * (self.current_num_observations/(self.current_num_observations+1))
        self.current_mean_returns = (self.current_num_observations*self.current_mean_returns + sample_observation) / (self.current_num_observations + 1)
        self.current_num_observations += 1

    # Function to take samples of our Normal-Inverse-Wishart distributions
    def _sample_niw(self, mean, cov, n_obs, num_samples):
        samples = []
        for _ in range(num_samples):
            sigma = invwishart.rvs(df=n_obs, scale=cov)
            mu = multivariate_normal.rvs(mean=mean, cov=sigma/n_obs)
            samples.append((mu, sigma))
        return samples


    def _kl_divergence(self, p_mu, p_sigma, q_mu, q_sigma):
        """KL divergence D_KL(P || Q) for two multivariate normals"""
        d = p_mu.shape[0]
        inv_q_sigma = np.linalg.inv(q_sigma)
        diff = q_mu - p_mu

        term1 = np.trace(inv_q_sigma @ p_sigma)
        term2 = diff.T @ inv_q_sigma @ diff
        term3 = np.log(np.linalg.det(q_sigma) / np.linalg.det(p_sigma))
        
        return 0.5 * (term1 + term2 - d + term3)

    # Employ JS-Divergence to determine the divergence between the current 
    def compute_js_divergence(self):
        # DO MONTE CARLO TO ESIMATE JS-DIVERGENCE 
        num_samples = 500
        
        # Using the same sample from mixture distribution for comparison between new and prev
        prev_dist_samples = self._sample_niw(self.previous_mean_returns, self.previous_covariance_returns, self.previous_num_observations, num_samples)
        curr_dist_samples = self._sample_niw(self.current_mean_returns, self.current_covariance_returns, self.current_num_observations, num_samples)
        mixture_dist_samples = []
        for _ in range(num_samples):
            if np.random.rand() < 0.5:
                mixture_dist_samples.append(self._sample_niw(self.previous_mean_returns, self.previous_covariance_returns, self.previous_num_observations, 1)[0])
            else:
                mixture_dist_samples.append(self._sample_niw(self.current_mean_returns, self.current_covariance_returns, self.current_num_observations, 1)[0])

        jsd_samples = []
        for i in range(len(mixture_dist_samples)):
            mu_p, cov_p = prev_dist_samples[i]
            mu_c, cov_c = curr_dist_samples[i]
            mu_m, cov_m = mixture_dist_samples[i]


            kl_pm = self._kl_divergence(mu_p, cov_p, mu_m, cov_m)
            kl_cm = self._kl_divergence(mu_c, cov_c, mu_m, cov_m)

            jsd_samples.append(0.5*(kl_pm+kl_cm))

        return np.mean(jsd_samples)

if __name__ == '__main__':
    opt = ConjugatePriorOptimizer(20, np.array([1, 2, 3]), np.array([[0.1,0.5,0.0],[0.5,0.2,0.0],[0.0,0.0,0.3]]))
    opt.update_prior(np.array([9, 10, 11]))
    print(opt.compute_js_divergence())