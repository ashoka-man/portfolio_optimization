# Automated Portfolio Rebalancer
Outline:
We will first construct an ideal portfolio in absence of a risk-free asset in accordance with Modern Portfolio Theory (Mean Variance given investor’s risk tolerance q) 
Then, we take into account the Sharpe Ratio and the Capital Allocation Line associated with a risk-free asset. We will maximize the Sharpe Ratio
We hope to improve the above models by considering rebalancing done by market signals/events rather than set time-periods. For example, using LSTMs for stock event prediction, HMM’s, etc., updating the prior return distribution based on Bayesian inference. 
For instance, we may use KL Divergence, to determine if a stock’s current behavior is drastically different from historical behavior

Iteration 1
The Model:
Each stock will have a return distribution characterized by a mean/variance. We will estimate the original prior parameters based on historical data. For the sake of simplicity with the model, we will use a log normal distribution to model absolute stock price or S_p \sim LN(\mu, \sigma^2) since stock prices are bounded by $0. Thus, if we model the returns as log returns, we will be able to produce an (ideally) normal distribution of returns over any period. This will be useful since our prior and posterior are from the same distribution. Our likelihood function, of course, is also normal.

With respect to Bayes rule, if a new set of evidence (a single day stock fall or rise) occurs with a low likelihood relative to our current stock weights, we update the portfolio based on the new evidence.

NOTE: This method is likely computationally expensive.

Bayesian Interpretation of Portfolio Returns:

P(R_p | D) = P(D | R_p) * P(R_p) / P(D)
Steps:

Likelihood: Takes the new data from the most recent time step and then considers the likelihood that our current portfolio’s return mean and variance would generate this datapoint. We will use a normal distribution as common for modelling log stock returns. We will also assume I.I.D between stock days (Random Walk) to simplify the calculations allowing for an iterative approach to updating the posterior. 

Prior: Models the current knowledge of the portfolio. We intend to use the  Normal-inverse-Wishart because it forms a conjugate pair with the Likelihood function.

Note that if we notice a big enough difference between the prior and posterior by some measure (probably KL Divergence), we will update our portfolio weights.  

Justification for Normal-inverse-Wishart distribution:
We are attempting to find the probability of new prices with respect to our accumulated distribution for securities prices and their covariances over the past time period. However, the true covariances between the securities in our portfolio are not a known constant.

Thus, we require an Inverse-Wishart distribution to create distribution for our risk, or our covariance. Furthermore, we need to estimate the probability of the new prices given the old prices with a normal distribution.

Multiplying these two distributions together provides us with the Normal-Inverse-Wishart distribution.
