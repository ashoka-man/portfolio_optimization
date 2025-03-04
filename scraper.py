import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class StockDataScraper:
    def __init__(self):
        pass
    
    def get_stock_data(self, stock_name, period_start, period_end):
        """
        Fetch historical stock data for a given stock symbol.
        """
        stock = yf.Ticker(stock_name)
        hist_data = stock.history(start=period_start, end=period_end)
        return hist_data
    
    def get_multiple_stocks(self, stock_list, period_start, period_end):
        """
        Fetch historical stock data for multiple stock symbols.
        """
        stock_data = {}
        for stock in stock_list:
            stock_data[stock] = self.get_stock_data(stock, period_start, period_end)
        return stock_data
    
    def get_financials(self, stock_name):
        """
        Fetch financial statements (income statement, balance sheet, cash flow) for a stock.
        """
        stock = yf.Ticker(stock_name)
        return {
            "Income Statement": stock.financials,
            "Balance Sheet": stock.balance_sheet,
            "Cash Flow": stock.cashflow
        }
    
    def get_key_stats(self, stock_name):
        """
        Fetch key statistics like market cap, PE ratio, dividend yield, etc.
        """
        stock = yf.Ticker(stock_name)
        return stock.info
    
    def plot_log_returns(self, stock_name, period):
        """
        Plot histogram of log returns over a specified period.
        """
        stock = yf.Ticker(stock_name)
        hist_data = stock.history(period=period)
        log_returns = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
        log_returns.dropna().plot.hist(bins=12, alpha=0.5, title=f'Log Returns of {stock_name}')
        plt.show()
    
if __name__ == "__main__":
    #quick test of functionality
    stocks = ["AAPL", "GOOGL", "TLSA", "AMZN"]
    start = "2023-01-01"
    end = "2024-01-01"
    scraper = StockDataScraper()
    stock_data_list = scraper.get_multiple_stocks(stocks, "2023-01-01", "2024-01-01")
    for name, stock in stock_data_list.items():
        print(name, stock['Close'][0], stock['Close'][-1])
    APPLE_BEGIN = 123.63253021240234
    APPLE_END = 191.38096618652344
    GOOGL_BEGIN = 88.79810333251953
    GOOGL_END = 139.18544006347656
    AMZN_BEGIN = 85.81999969482422
    AMZN_END = 151.94000244140625
    TSLA_BEGIN = 0.5799999833106995
    TSLA_END = 0.5600000023841858

    PORTFOLIO_BEGIN = APPLE_BEGIN * 0.8007794044414817 + GOOGL_BEGIN * 0.014117660860342551 + AMZN_BEGIN * 0.18951722774849417 + TSLA_BEGIN * -0.004414293050318417
    PORTFOLIO_END = APPLE_END * 0.8007794044414817 + GOOGL_END * 0.014117660860342551 + AMZN_END * 0.18951722774849417 + TSLA_END * -0.004414293050318417
    print((PORTFOLIO_END - PORTFOLIO_BEGIN) / PORTFOLIO_BEGIN)
    # scraper = StockDataScraper()
    # stock_data = scraper.get_stock_data("MSFT", "2023-01-01", "2024-01-01")
    # financials = scraper.get_financials("MSFT")
    # key_stats = scraper.get_key_stats("MSFT")
    # scraper.plot_log_returns("MSFT", "6mo")
    
    # print(stock_data.head())
    # print(financials)
    # print(key_stats)
