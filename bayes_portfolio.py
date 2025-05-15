from bayesian import ConjugatePriorOptimizer
from scraper import StockDataScraper
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class Portfolio():
    def __init__(self, start_date, fitting_end, end_date, stocks, initial_allocations):
        self.scraper = StockDataScraper()

        self.fitting_start = start_date
        self.fitting_end = fitting_end
        self.processing_start = (datetime.strptime(fitting_end, "%Y-%m-%d") 
                                 + timedelta(days = 1)).strftime("%Y-%m-%d")
        self.processing_end = end_date

        # Initialize the portfolio based on the amount of each stock we purchase, as well as the proportions they take up of the full portfolio
        self.stock_list = stocks
        self.abs_portfolio = np.array(initial_allocations)
        self.prop_portfolio = (1/sum(self.abs_portfolio)) * self.abs_portfolio

        # Initializing absolute and proportional data sets for the fitting and the processing data
        self.abs_data = self._compute_port_data(self.fitting_start, self.fitting_end)
        self.prop_data = self._compute_prop_data(self.abs_data)
        self.abs_processing = self._compute_port_data(self.processing_start, self.processing_end)
        self.prop_processing = self._compute_prop_data(self.abs_processing)

        # Computing NIW parameters for absolute returns and proportions
        self.returns = self._compute_abs_returns()
        self.mean_returns = self.returns.mean()
        self.cov_returns = self.returns.cov()

        self.mean_prop = self.prop_data.mean()
        self.cov_prop = self.prop_data.cov()
        
        # Initialize optimizer
        self.optimizer = ConjugatePriorOptimizer(self.abs_data.shape[0], self.mean_returns, self.cov_returns, self.mean_prop, self.cov_prop)
    
    # Standard data fetching, SOMETHING IS WRONG WITH YFINANCE
    def _fetch_data(self, start, end):
        """Fetch historical stock prices for given stocks."""
        stock_data = self.scraper.get_multiple_stocks(self.stock_list, start, end)
        prices = pd.DataFrame({stock: stock_data[stock]['Close'] for stock in self.stock_list})
        return prices
    
    # Scaling the dataframe with respect to the allocation of each stock (absolute values)
    def _compute_port_data(self, start, end):
        data = self._fetch_data(start, end)
        return pd.DataFrame({self.stock_list[i] : 
                             (self.abs_portfolio[i]/data.iloc[0][i]) * data[self.stock_list[i]] 
                             for i in range(len(self.stock_list))})
    
    # Normalizing the portfolio with respect to the proportion each stock takes up
    def _compute_prop_data(self, data : pd.DataFrame):
        normalize_consts = data.sum(axis = 1)
        return data.div(normalize_consts, axis = 0)

    def _compute_abs_returns(self):
        """Calculate simple returns of stock prices."""
        return (self.abs_data / self.abs_data.shift(1) - 1).dropna()
    
if __name__ == "__main__":
    stocks = ["MSFT", "NVDA", "AMZN", "KO"]
    initial_alloc = [100, 150, 50, 200]
    start = "2023-01-01"
    fitting_end = "2023-01-05"
    end = "2024-01-01"

    portfolio = Portfolio(start, fitting_end, end, stocks, initial_alloc)
    portfolio.optimizer.update_prior(np.array([4, 10, 20]))
    print(portfolio.optimizer.compute_total_js_divergence())