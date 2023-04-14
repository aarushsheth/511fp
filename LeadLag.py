from datetime import datetime
import pandas as pd
import numpy as np
import numba
from numba import njit
from numba.typed import Dict
from numba.typed import List
import networkx as nx
import pickle
import matplotlib.pyplot as plt

@njit
def process_pair(pairValues, n, x, total_pairs):
    m = np.zeros((498, 498))
    for index in range(total_pairs):
        i = x[index // n]
        j = x[index % n]
        x1 = x.index(i)
        y1 = x.index(j)
        pairValuesArray = np.array(
            [pairValues[f"{i}r"], pairValues[f"{j}ryi"],
             pairValues[f"{j}rys"]])
        if (np.isnan(pairValuesArray[0]) or np.isnan(pairValuesArray[1])
            or np.isnan(pairValuesArray[2])):
            continue
        if (pairValuesArray[0] >= 0):
            if pairValuesArray[0] <= pairValuesArray[1] and pairValuesArray[0] <= pairValuesArray[2]:
                m[x1][y1] = 1
        if (pairValuesArray[0] < 0):
            if pairValuesArray[0] <= pairValuesArray[2] and pairValuesArray[0] <= pairValuesArray[1]:
                m[x1][y1] = 1
    return m

def process_month(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    results = []
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        filtered_data = stockData[stockData["datadate"] == datetime.strptime(
            date_str, "%Y-%m-%d")]
        if not filtered_data.empty:
            print(f"Non-empty data found for {date_str}")
            filtered_data = filtered_data.drop("datadate", axis=1)
            filtered_data = filtered_data.drop("Unnamed: 0", axis=1)
            data_dict = filtered_data.to_dict()
            typed_dict = Dict.empty(
                key_type=numba.types.unicode_type,
                value_type=numba.types.float64)
            for key, value in data_dict.items():
                typed_dict[key] = value[next(iter(value))]
            result = process_pair(typed_dict, n, typed_x, total_pairs)
            results.append(result)
    return np.array(results)

def build_graph(adj_matrix, stock_symbols):
    G = nx.DiGraph()
    for i, stock_i in enumerate(stock_symbols):
        for j, stock_j in enumerate(stock_symbols):
            weight = adj_matrix[i][j]
            if weight > 0:
                G.add_edge(stock_i, stock_j, weight=weight)     
    return G

def visualize_graph(G, node_colors):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, font_weight='bold', node_color=node_colors, alpha=0.8, arrowsize=12)
    plt.show()

def modularity(G):
    communities = nx.algorithms.community.greedy_modularity_communities(G.to_undirected())
    modularity = nx.algorithms.community.modularity(G.to_undirected(), communities)
    return modularity

def eigenvector_centrality(G):
    eigenvector_centralities = nx.eigenvector_centrality(G)
    return eigenvector_centralities

def diameter(G):
    if nx.is_connected(G.to_undirected()):
        return nx.diameter(G.to_undirected())
    else:
        return float('inf')

def average_path_length(G):
    if nx.is_connected(G.to_undirected()):
        return nx.average_shortest_path_length(G.to_undirected())
    else:
        return float('inf')

def average_node_degree(G):
    return 2 * G.number_of_edges() / G.number_of_nodes()

def clustering_coefficients(G):
    return nx.clustering(G.to_undirected())

def portfolio_value(portfolio, stock_prices):
    return sum(shares * stock_prices[stock] for stock, shares in portfolio.items())

def buy_stock(portfolio, stock, amount, price):
    portfolio[stock] += amount / price

def sell_stock(portfolio, stock, amount, price):
    portfolio[stock] -= amount / price

# start_time = datetime.now()
stockData = pd.read_csv("finally.csv")
stockData['datadate'] = pd.to_datetime(stockData['datadate'])
stockData = stockData.sort_values(by='datadate')
with open("newstocks.txt", "r") as f: x = f.read().splitlines()
x[0] = "A"
n = len(x)
total_pairs = n * n
typed_x = List(x)
start_date = '2013-03-15'
end_date = '2023-03-15'
# month = process_month(start_date, end_date)
# with open('month_data.pkl', 'wb') as f:
#     pickle.dump(month_data, f)
# end_time = datetime.now()
# print('Computations completed, duration: {}'.format(end_time - start_time))
with open('day_data.pkl', 'rb') as f: loaded_daily_data = pickle.load(f)
with open('weekly_data.pkl', 'rb') as f: loaded_weekly_data = pickle.load(f)
with open('monthly_data.pkl', 'rb') as f: loaded_monthly_data = pickle.load(f)
summed_data = np.sum(loaded_daily_data[:2018, :, :], axis=0)
masked_array = np.ma.masked_array(summed_data, mask=np.eye(498, dtype=bool))
indices_flattened = np.argpartition(masked_array.ravel(), -10)[-10:]
row_indices, col_indices = np.unravel_index(indices_flattened, masked_array.shape)
with open("newstocks.txt", "r") as f: stock_names = f.read().splitlines()
# for row, col in zip(row_indices, col_indices):
#     print(f"Row: {row} ({stock_names[row]}), Col: {col} ({stock_names[col]}). Sum value: {summed_data[row, col]}")
leader_lagger_dict = {stock_names[row]: stock_names[col] for row, col in zip(row_indices, col_indices)}
# for leader, lagger in leader_lagger_dict.items():
#     print(f"Leader: {leader}, Lagger: {lagger}")
twoyears_daily_data = loaded_daily_data[-500:, :, :]
stocks_to_include = set(leader_lagger_dict.keys()).union(set(leader_lagger_dict.values()))
filtered_stock_data = stockData.loc[:, stocks_to_include].tail(500)
initial_investment = 500000
n_laggers = len(leader_lagger_dict.values())
investment_per_stock = initial_investment / n_laggers
portfolio = {stock: 0 for stock in leader_lagger_dict.values()}

for stock in portfolio:
    price = filtered_stock_data[stock].iloc[0]
    buy_stock(portfolio, stock, investment_per_stock, price)

n_days = filtered_stock_data.shape[0]
trailing_stop_percentage = 0.02

highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
for leader, lagger in leader_lagger_dict.items():
    for i in range(n_days):
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
                    sell_stock(portfolio, lagger, investment_per_stock, price)
                    break
            break

rounded_portfolio = {k: round(v, 2) for k, v in portfolio.items()}
print(rounded_portfolio)
portfolio_values = []

for day in range(n_days):
    stock_prices = filtered_stock_data.iloc[day]
    current_value = portfolio_value(portfolio, stock_prices)
    portfolio_values.append(current_value)
    
initial_value = initial_investment
final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1])
portfolio_return = (final_value - initial_value) / initial_value
print(f"Portfolio return over the period: {portfolio_return * 100:.2f}%")

daily_portfolio_values = []
for day in range(n_days):
    stock_prices = filtered_stock_data.iloc[day]
    current_value = portfolio_value(portfolio, stock_prices)
    daily_portfolio_values.append(current_value)

fig, ax1 = plt.subplots()
ax1.plot(range(n_days), daily_portfolio_values, color='blue')
ax1.set_xlabel("Days")
ax1.set_ylabel("Portfolio Value", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(range(n_days), [100 * (value - initial_value) / initial_value for value in daily_portfolio_values], color='red')
ax2.set_ylabel("Portfolio Return (%)", color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.title("Portfolio Value and Return Over Time")
plt.show()

# leader_stocks = list(leader_lagger_dict.keys())
# lagger_stocks = list(leader_lagger_dict.values())
# stock_colors = ['red' if stock in leader_stocks else 'blue' for stock in stocks_to_include]
# adj_matrix = filtered_stock_data[0].copy()
# G = build_graph(adj_matrix, stocks_to_include)
# visualize_graph(G, stock_colors)
# print("Modularity:", round(modularity(G),2))
# eigenvector_centralities = eigenvector_centrality(G)
# rounded_eigenvector_centralities = {k: round(v, 2) for k, v in eigenvector_centralities.items()}
# print("Eigenvector Centralities:", rounded_eigenvector_centralities)
# print("Diameter:", round(diameter(G),2))
# print("Average Path Length:", round(average_path_length(G),2))
# print("Average Node Degree:", round(average_node_degree(G),2))
# clustering_coeffs = clustering_coefficients(G)
# rounded_clustering_coeffs = {k: round(v, 2) for k, v in clustering_coeffs.items()}
# print("Clustering Coefficients:", rounded_clustering_coeffs)
