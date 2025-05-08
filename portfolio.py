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
        self.risk_free_rate = (1 + risk_free_rate)**(1/360) - 1 #save to be daily
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data() #Daily stock price for each stock 
        self.returns = self._compute_simple_returns() # in simple
        self.mean_returns = self.returns.mean() # in sim
        self.cov_matrix = self.returns.cov() # covariance between log returns

    def _fetch_data(self):
        """Fetch historical stock prices for given stocks."""
        stock_data = self.scraper.get_multiple_stocks(self.stock_list, self.start_date, self.end_date)
        prices = pd.DataFrame({stock: stock_data[stock]['Close'] for stock in self.stock_list})
        return prices

    def _compute_log_returns(self):
        """Calculate log returns of stock prices."""
        return np.log(self.data / self.data.shift(1)).dropna()


    def _compute_simple_returns(self):
        """Calculate simple returns of stock prices."""
        return (self.data / self.data.shift(1) - 1).dropna()


    def _portfolio_performance(self, weights):
        """Calculate portfolio return and volatility."""
        weights = np.array(weights)        
        port_return = (weights.T @ self.mean_returns)  # Expected portfolio return
        port_volatility = weights.T @ (self.cov_matrix @ weights) # Portfolio volatility

        return port_return, port_volatility
    
    def portfolio_graph(self, weight, initial_investment=100):
        '''
        show portfolio graph over time
        '''
        weight_array = np.array(weight)

        # Apply weights to stock prices correctly
        weighted_prices = self.data.mul(weight_array, axis=1)
        
        # Compute initial portfolio value
        initial_prices = weighted_prices.iloc[0].sum()
        
        # Compute portfolio values over time
        portfolio_values = weighted_prices.sum(axis=1) * initial_investment / initial_prices

        # Plot each stock's weighted contribution
        plt.figure(figsize=(12, 6))
        for stock in self.stock_list:
            plt.plot(self.data.index, weighted_prices[stock], label=f"{stock} (Weighted)")

        # Plot total portfolio value
        plt.plot(self.data.index, portfolio_values, label="Portfolio Value", linewidth=2, color="black")

        # Formatting
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Portfolio Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()


    def optimize_mean_variance_portfolio_with_no_risk_free_asset(self, q):
        """
        Find optimal portfolio weights using mean-variance optimization.
        q is the risk rate from [1, infinity]
        """
        assert(q >= 0)
        num_assets = len(self.stock_list)
        one_vector = np.ones(num_assets)
        inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        one_vector_inv_cov_matrix = one_vector.T @ inv_cov_matrix 
        lambda_lagrange = ((q * one_vector_inv_cov_matrix @ self.mean_returns) - 2)/ (one_vector_inv_cov_matrix @ one_vector)
        w = 0.5 * inv_cov_matrix @ (q*self.mean_returns - lambda_lagrange * one_vector)
        
        return w
    

    def optimize_mean_variance_portfolio_with_risk_free_asset(self):
        """
        Find optimal portfolio weights using mean-variance optimization.
        q is the risk rate from [1, infinity]
        """

        def neg_sharpe_ratio(weights):
            '''
            Optimize for negative sharp ratio
            '''
            port_return, port_volatility = self._portfolio_performance(weights)
            return -(port_return - self.risk_free_rate) / np.sqrt(port_volatility)
        w0 = self.optimize_mean_variance_portfolio_with_no_risk_free_asset(0)

        w_opt = minimize(
            neg_sharpe_ratio, 
            x0=w0, 
            bounds=[(0,1) for _ in range(len(w0))], 
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1} 
        )
        return w_opt.x

    def plot_efficient_frontier(self, num_portfolios=10000, optimal_weights=None):
        """Plot the Efficient Frontier with the optimal portfolio highlighted."""
        num_assets = len(self.stock_list)
        
        results = np.zeros((3, num_portfolios))  # Store returns, volatility, and Sharpe ratios
        weights_list = []

        for i in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(num_assets))  # Generate random weights
            weights_list.append(weights)
            port_return, port_volatility = self._portfolio_performance(weights)
            sharpe_ratio = (port_return - self.risk_free_rate) / np.sqrt(port_volatility)
            
            results[0, i] = port_return
            results[1, i] = np.sqrt(port_volatility)
            results[2, i] = sharpe_ratio

        #np.exp(returns) - 1 to take back to regular daily return
        plt.scatter(results[1, :], np.exp(results[0, :])-1, c=results[2, :], cmap="viridis", marker="o", alpha=0.5, label="Random Portfolios")
        
        if optimal_weights is not None:
            for weight in optimal_weights:
                opt_return, opt_volatility = self._portfolio_performance(weight)
                plt.scatter(np.sqrt(opt_volatility), np.exp(opt_return)-1, color='red', marker='*', s=200, label='Optimal Portfolio')
        
        plt.xlabel("Volatility (Standard Deviation)")
        plt.ylabel("Expected Simple Return")
        plt.title("Efficient Frontier")
        plt.colorbar(label="Sharpe Ratio")
        plt.legend()
        plt.show()

    def graph_correlation_matrix(self):
        # Plot covariance matrix using Matplotlib
        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.imshow(self.cov_matrix, cmap='coolwarm', interpolation='nearest')

        # Add colorbar
        cbar = fig.colorbar(cax)
        cbar.set_label("Covariance")

        # Set axis labels and ticks
        ax.set_xticks(np.arange(len(self.stock_list)))
        ax.set_yticks(np.arange(len(self.stock_list)))
        ax.set_xticklabels(self.stock_list, rotation=45)
        ax.set_yticklabels(self.stock_list)

        # Add title
        plt.title("Asset Covariance Matrix")

        # Show the plot
        plt.show()

# Example Usage
if __name__ == "__main__":
    stocks = ["MSFT", "NVDA", "AMZN", "KO"]
    start = "2023-01-01"
    end = "2024-01-01"
    
    optimizer = PortfolioOptimizer(stocks, start, end, risk_free_rate=0.0433)
    optimal_weights = [optimizer.optimize_mean_variance_portfolio_with_no_risk_free_asset(0), optimizer.optimize_mean_variance_portfolio_with_risk_free_asset()]
    print(f"Optimal Portfolio Weights: {dict(zip(stocks, optimal_weights))}")
    # daily_log_returns, daily_volatility = optimizer._portfolio_performance(optimal_weights)

    # # Convert to annualized returns
    # annual_log_return = daily_log_returns * 252
    # annual_volatility = daily_volatility * 252

    # annual_simple_return = np.exp(annual_log_return) - 1
    # print(annual_simple_return, annual_volatility)

    # optimizer.plot_efficient_frontier(optimal_weights=optimal_weights)
    optimizer.plot_efficient_frontier(optimal_weights=optimal_weights)
    # print(optimizer.portfolio_graph(optimal_weights))
    # optimizer.graph_correlation_matrix()
    print(optimizer.cov_matrix)