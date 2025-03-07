from portfolio import PortfolioOptimizer
from scraper import StockDataScraper

class Backtester():
    def __init__(start_date, training_end, end_date):
        scraper = StockDataScraper()
        