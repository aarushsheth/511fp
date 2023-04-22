
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf


def buy_stock(portfolio, stock, amount, price):
    portfolio[stock] += amount / price


def sell_stock(portfolio, stock, amount, price):
    portfolio[stock] -= amount / price


def portfolio_value(portfolio, stock_prices):
    return sum(shares * stock_prices[stock] for stock, shares in portfolio.items())


def average_out_degree(adj_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    out_degrees = dict(G.out_degree())
    average_out_degrees = {k: v / len(G) for k, v in out_degrees.items()}
    return average_out_degrees


def simulate_trading(trailing_stop_percentage):
    remaining_cash = initial_investment
    portfolio = {stock: 0 for stock in leader_lagger_dict.values()}
    highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
    portfolio_cash = {stock: 0 for stock in leader_lagger_dict.values()}
    for i in range(1, n_days):
        for leader, lagger in leader_lagger_dict.items():
            current_leader_price = filtered_stock_data[leader].iloc[i]
            highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
            if current_leader_price >= 1.08 * filtered_stock_data[leader].iloc[i - 1]:
                price = filtered_stock_data[lagger].iloc[i]
                cash_to_invest = remaining_cash / n_laggers
                buy_stock(portfolio, lagger, cash_to_invest, price)
                portfolio_cash[lagger] += cash_to_invest
                remaining_cash -= cash_to_invest
            current_leader_price = filtered_stock_data[leader].iloc[i]
            if current_leader_price <= (trailing_stop_percentage) * highest_leader_prices[leader]:
                price = filtered_stock_data[lagger].iloc[i]
                sell_stock(portfolio, lagger, portfolio[lagger], price)
    initial_value = initial_investment
    final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1]) + remaining_cash
    portfolio_return = (final_value - initial_value) / initial_value
    return portfolio_return


with open('day_data.pkl', 'rb') as f:
    loaded_daily_data = pickle.load(f)

with open("newstocks.txt", "r") as f:
    stock_names = f.read().splitlines()

stockData = pd.read_csv("finally.csv")
stockData['datadate'] = pd.to_datetime(stockData['datadate'])
stockData = stockData.sort_values(by='datadate')
stockData.set_index('datadate', inplace=True)

summed_data = np.sum(loaded_daily_data[:2268, :, :], axis=0)
masked_array = np.ma.masked_array(summed_data, mask=np.eye(498, dtype=bool))
indices_flattened = np.argpartition(masked_array.ravel(), -10)[-10:]
row_indices, col_indices = np.unravel_index(indices_flattened, masked_array.shape)
leader_lagger_dict = {stock_names[row]: stock_names[col] for row, col in zip(row_indices, col_indices)}

year_daily_data = loaded_daily_data[-250:, :, :]
pseudo_adj_matrix = year_daily_data[0]
avg_out_degrees = average_out_degree(pseudo_adj_matrix)

stocks_to_include = set(leader_lagger_dict.keys()).union(set(leader_lagger_dict.values()))
filtered_stock_data = stockData.loc[:, stocks_to_include].tail(250)

n_laggers = len(leader_lagger_dict.values())
initial_investment = 500000
n_days = filtered_stock_data.shape[0]

trailing_stop_percentages = np.linspace(0, 1, 101)
portfolio_returns = []

for trailing_stop_percentage in trailing_stop_percentages:
    portfolio_return = simulate_trading(trailing_stop_percentage)
    portfolio_returns.append(portfolio_return)

plt.plot(trailing_stop_percentages, portfolio_returns)
plt.xlabel("Trailing Stop Percentage")
plt.ylabel("Portfolio Return")
plt.title("Portfolio Return vs. Trailing Stop Percentage")
plt.show()

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import yfinance as yf


def buy_stock(portfolio, stock, amount, price):
    portfolio[stock] += amount / price


def sell_stock(portfolio, stock, amount, price):
    portfolio[stock] -= amount / price


def portfolio_value(portfolio, stock_prices):
    return sum(shares * stock_prices[stock] for stock, shares in portfolio.items())


def average_out_degree(adj_matrix):
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    out_degrees = dict(G.out_degree())
    average_out_degrees = {k: v / len(G) for k, v in out_degrees.items()}
    return average_out_degrees


def simulate_trading(trailing_stop_percentage):
    remaining_cash = initial_investment
    portfolio = {stock: 0 for stock in leader_lagger_dict.values()}
    highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
    portfolio_cash = {stock: 0 for stock in leader_lagger_dict.values()}
    for i in range(1, n_days):
        for leader, lagger in leader_lagger_dict.items():
            current_leader_price = filtered_stock_data[leader].iloc[i]
            highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
            if current_leader_price >= 1.08 * filtered_stock_data[leader].iloc[i - 1]:
                price = filtered_stock_data[lagger].iloc[i]
                cash_to_invest = remaining_cash / n_laggers
                buy_stock(portfolio, lagger, cash_to_invest, price)
                portfolio_cash[lagger] += cash_to_invest
                remaining_cash -= cash_to_invest
            current_leader_price = filtered_stock_data[leader].iloc[i]
            if current_leader_price <= (trailing_stop_percentage) * highest_leader_prices[leader]:
                price = filtered_stock_data[lagger].iloc[i]
                sell_stock(portfolio, lagger, portfolio[lagger], price)
    initial_value = initial_investment
    final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1]) + remaining_cash
    portfolio_return = (final_value - initial_value) / initial_value
    return portfolio_return


with open('day_data.pkl', 'rb') as f:
    loaded_daily_data = pickle.load(f)

with open("newstocks.txt", "r") as f:
    stock_names = f.read().splitlines()

stockData = pd.read_csv("finally.csv")
stockData['datadate'] = pd.to_datetime(stockData['datadate'])
stockData = stockData.sort_values(by='datadate')
stockData.set_index('datadate', inplace=True)

summed_data = np.sum(loaded_daily_data[:2268, :, :], axis=0)
masked_array = np.ma.masked_array(summed_data, mask=np.eye(498, dtype=bool))
indices_flattened = np.argpartition(masked_array.ravel(), -10)[-10:]
row_indices, col_indices = np.unravel_index(indices_flattened, masked_array.shape)
leader_lagger_dict = {stock_names[row]: stock_names[col] for row, col in zip(row_indices, col_indices)}

year_daily_data = loaded_daily_data[-250:, :, :]
pseudo_adj_matrix = year_daily_data[0]
avg_out_degrees = average_out_degree(pseudo_adj_matrix)

stocks_to_include = set(leader_lagger_dict.keys()).union(set(leader_lagger_dict.values()))
filtered_stock_data = stockData.loc[:, stocks_to_include].tail(250)

n_laggers = len(leader_lagger_dict.values())
initial_investment = 500000
n_days = filtered_stock_data.shape[0]

trailing_stop_percentages = np.linspace(0, 1, 101)
portfolio_returns = []

for trailing_stop_percentage in trailing_stop_percentages:
    portfolio_return = simulate_trading(trailing_stop_percentage)
    portfolio_returns.append(portfolio_return)

plt.plot(trailing_stop_percentages, portfolio_returns)
plt.xlabel("Trailing Stop Percentage")
plt.ylabel("Portfolio Return")
plt.title("Portfolio Return vs. Trailing Stop Percentage")
plt.show()
