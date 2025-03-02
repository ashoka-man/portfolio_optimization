import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

class DataScraper:
    def __init__():
        pass

    def get_stock(self, stock_name, period_start, period_end):
        pass

    def get_collection(self, stock_list, period_start, period_end):
        pass


if __name__ == "__main__":
    dat = yf.Ticker("MSFT")
    one_mo_hist = dat.history(period = '6mo')[['Open', 'High', 'Low', 'Close']]
    log_returns = np.log(one_mo_hist['Close'] / one_mo_hist['Close'].shift(1))
    # scaled_vol = (one_mo_hist['High'] - one_mo_hist['Low'])/one_mo_hist['Low']
    # Plot the histogram of log returns
    plot = log_returns.dropna().plot.hist(bins=12, alpha=0.5)
    plot.plot()
    # Print the log returns
    print(log_returns)
    
    # Show the plot
    plt.show()
