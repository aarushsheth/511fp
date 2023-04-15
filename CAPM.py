import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import csv

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def compute_CAPM(stock_prices, market_prices):
    aligned_data = pd.concat([stock_prices, market_prices], axis=1).dropna()
    stock_returns = aligned_data.iloc[:, 0].pct_change().dropna()
    market_returns = aligned_data.iloc[:, 1].pct_change().dropna()

    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]

    risk_free_rate = 0.0089
    market_return = market_returns.mean() * 252

    CAPM = risk_free_rate + beta * (market_return - risk_free_rate)
    return CAPM

def main():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)

    with open("newstocks.txt", "r") as f:
        stock_tickers = [ticker.strip() for ticker in f.readlines()]

    market_ticker = "^GSPC"
    market_prices = get_stock_data(market_ticker, start_date, end_date)["Adj Close"]

    with open("capm_values.csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Ticker", "CAPM"])

        for ticker in stock_tickers:
            stock_prices = get_stock_data(ticker, start_date, end_date)["Adj Close"]
            CAPM = compute_CAPM(stock_prices, market_prices)
            print(f"CAPM for {ticker}: {CAPM}")

            csv_writer.writerow([ticker, CAPM])

if __name__ == "__main__":
    main()