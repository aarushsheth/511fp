import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import networkx as nx

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

with open('day_data.pkl', 'rb') as f:
    loaded_daily_data = pickle.load(f)
with open('weekly_data.pkl', 'rb') as f:
    loaded_weekly_data = pickle.load(f)
with open('monthly_data.pkl', 'rb') as f:
    loaded_monthly_data = pickle.load(f)
with open("newstocks.txt", "r") as f:
    stock_names = f.read().splitlines()

stockData = pd.read_csv("finally.csv")
stockData['datadate'] = pd.to_datetime(stockData['datadate'])
stockData = stockData.sort_values(by='datadate')

summed_data = np.sum(loaded_daily_data[:2468, :, :], axis=0)
masked_array = np.ma.masked_array(summed_data, mask=np.eye(498, dtype=bool))
indices_flattened = np.argpartition(masked_array.ravel(), -10)[-10:]
row_indices, col_indices = np.unravel_index(indices_flattened, masked_array.shape)
leader_lagger_dict = {stock_names[row]: stock_names[col] for row, col in zip(row_indices, col_indices)}

year_daily_data = loaded_daily_data[-50:, :, :]
pseudo_adj_matrix = year_daily_data[0]
avg_out_degrees = average_out_degree(pseudo_adj_matrix)
stocks_to_include = set(leader_lagger_dict.keys()).union(set(leader_lagger_dict.values()))
filtered_stock_data = stockData.loc[:, stocks_to_include].tail(50)

n_laggers = len(leader_lagger_dict.values())
initial_investment = 500000
n_days = filtered_stock_data.shape[0]

portfolio = {stock: 0 for stock in leader_lagger_dict.values()}

trailing_stop_percentage = 0.10

for leader, lagger in leader_lagger_dict.items():
    investment_per_stock = initial_investment / n_laggers
    highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
    
    for i in range(1, n_days):
        current_leader_price = filtered_stock_data[leader].iloc[i]
        highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
        
        if current_leader_price >= 1.02 * filtered_stock_data[leader].iloc[i - 1]:
            price = filtered_stock_data[lagger].iloc[i]
            buy_stock(portfolio, lagger, investment_per_stock, price)
            
            for j in range(i, n_days):
                current_leader_price = filtered_stock_data[leader].iloc[j]
                highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
                
                if current_leader_price <= (1 - trailing_stop_percentage) * highest_leader_prices[leader]:
                    price = filtered_stock_data[lagger].iloc[j]
                    sell_stock(portfolio, lagger, portfolio[lagger] * price, price)
                    break
                    
            break

rounded_portfolio = {k: round(v, 2) for k, v in portfolio.items()}
print(rounded_portfolio)
portfolio_values = [initial_investment]

for day in range(1, n_days):
    stock_prices = filtered_stock_data.iloc[day]
    current_value = portfolio_value(portfolio, stock_prices)
    portfolio_values.append(current_value)

initial_value = initial_investment
final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1])
portfolio_return = (final_value - initial_value) / initial_value
print(f"Portfolio return over the period: {portfolio_return * 100:.2f}%")

fig, ax1 = plt.subplots()
ax1.plot(range(n_days), portfolio_values, color='blue')
ax1.set_xlabel("Days")
ax1.set_ylabel("Portfolio Value", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(range(n_days), [100 * (value - initial_value) / initial_value for value in portfolio_values], color='red')
ax2.set_ylabel("Portfolio Return (%)", color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title("Portfolio Value and Return Over Time")
plt.show()
