import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scraper import StockDataScraper

class PortfolioOptimizer:
    def __init__(self, stock_list, start_date, end_date, risk_free_rate=0.0):
        self.scraper = StockDataScraper()
        self.stock_list = stock_list
        self.risk_free_rate = risk_free_rate
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()
        self.returns = self._compute_log_returns()

    def _fetch_data(self):
        """Fetch historical stock prices for given stocks."""
        stock_data = self.scraper.get_multiple_stocks(self.stock_list, self.start_date, self.end_date)
        prices = pd.DataFrame({stock: stock_data[stock]['Close'] for stock in self.stock_list})
        return prices

    def _compute_log_returns(self):
        """Calculate log returns of stock prices."""
        return np.log(self.data / self.data.shift(1)).dropna()

    def _portfolio_performance(self, weights):
        """Calculate portfolio return and volatility."""
        weights = np.array(weights)
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()
        
        port_return = np.dot(weights, mean_returns)  # Expected portfolio return
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio volatility

        return port_return, port_volatility

    def optimize_portfolio(self):
        """Find optimal portfolio weights using mean-variance optimization."""
        num_assets = len(self.stock_list)
        init_guess = np.ones(num_assets) / num_assets  # Equal weighting
        bounds = [(0, 1) for _ in range(num_assets)]  # No short-selling
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights = 1

        def neg_sharpe_ratio(weights):
            """Objective function: minimize negative Sharpe Ratio (maximize Sharpe)."""
            ret, vol = self._portfolio_performance(weights)
            return -ret / vol  # Assuming risk-free rate = 0 for simplicity

        result = minimize(neg_sharpe_ratio, init_guess, bounds=bounds, constraints=constraints)
        return result.x  # Optimal portfolio weights

    def plot_efficient_frontier(self, num_portfolios=10000, optimal_weights=None):
        """Plot the Efficient Frontier with the optimal portfolio highlighted."""
        num_assets = len(self.stock_list)
        
        results = np.zeros((3, num_portfolios))  # Store returns, volatility, and Sharpe ratios
        weights_list = []

        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(num_assets))  # Generate random weights
            weights_list.append(weights)
            port_return, port_volatility = self._portfolio_performance(weights)
            sharpe_ratio = port_return / port_volatility
            
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = sharpe_ratio

        #np.exp(returns) - 1 to take back to regular daily return
        plt.scatter(results[1, :], np.exp(results[0, :])-1, c=results[2, :], cmap="viridis", marker="o", alpha=0.5, label="Random Portfolios")
        
        if optimal_weights is not None:
            opt_return, opt_volatility = self._portfolio_performance(optimal_weights)
            plt.scatter(opt_volatility, np.exp(opt_return)-1, color='red', marker='*', s=200, label='Optimal Portfolio')
        
        plt.xlabel("Volatility (Risk)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.colorbar(label="Sharpe Ratio")
        plt.legend()
        plt.show()

# Example Usage
if __name__ == "__main__":
    stocks = ["AAPL", "GOOGL", "TLSA", "AMZN"]
    start = "2023-01-01"
    end = "2024-01-01"
    
    optimizer = PortfolioOptimizer(stocks, start, end)
    optimal_weights = optimizer.optimize_portfolio()
    print(f"Optimal Portfolio Weights: {dict(zip(stocks, optimal_weights))}")
    daily_log_returns, daily_volatility = optimizer._portfolio_performance(optimal_weights)

    # Convert to annualized returns
    annual_log_return = daily_log_returns * 252
    annual_volatility = daily_volatility * np.sqrt(252)

    annual_simple_return = np.exp(annual_log_return) - 1
    print(annual_simple_return, annual_volatility)

    optimizer.plot_efficient_frontier(optimal_weights=optimal_weights)
