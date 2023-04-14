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
investment_per_stock = initial_investment / n_laggers
portfolio = {stock: 0 for stock in leader_lagger_dict.values()}

trailing_stop_percentage = 0.10
highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
remaining_cash = initial_investment
portfolio_cash = {stock: 0 for stock in leader_lagger_dict.values()}

for i in range(1, n_days):
    for leader, lagger in leader_lagger_dict.items():
        current_leader_price = filtered_stock_data[leader].iloc[i]
        highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
        
        if current_leader_price >= 1.02 * filtered_stock_data[leader].iloc[i - 1]:
            price = filtered_stock_data[lagger].iloc[i]
            cash_to_invest = remaining_cash / n_laggers
            buy_stock(portfolio, lagger, cash_to_invest, price)
            portfolio_cash[lagger] += cash_to_invest
            remaining_cash -= cash_to_invest
        
        current_leader_price = filtered_stock_data[leader].iloc[i]
        if current_leader_price <= (1 - trailing_stop_percentage) * highest_leader_prices[leader]:
            price = filtered_stock_data[lagger].iloc[i]
            sell_stock(portfolio, lagger, portfolio[lagger] * price, price)
            cash_to_receive = portfolio[lagger] * price
            portfolio_cash[lagger] -= cash_to_receive
            remaining_cash += cash_to_receive

rounded_portfolio = {k: round(v, 2) for k, v in portfolio.items()}
print(rounded_portfolio)
initial_value = initial_investment
final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1]) + remaining_cash
portfolio_return = (final_value - initial_value) / initial_value
print(f"Portfolio return over the period: {portfolio_return * 100:.2f}%")

stock_prices_list = filtered_stock_data.values.T
portfolio_values = [portfolio_value(portfolio, filtered_stock_data.iloc[i]) + remaining_cash for i in range(n_days)]

fig, ax1 = plt.subplots()
ax1.plot(range(n_days), portfolio_values, color='blue')
ax1.set_xlabel("Days")
ax1.set_ylabel("Portfolio Value", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax2 = ax1.twinx()
ax2.plot(range(n_days), [100 * (value - initial_value) / initial_value for value in portfolio_values], color='red')
ax2.set_ylabel("Portfolio Return (%)", color='red')
plt.title("Portfolio Value and Return Over Time")
plt.show()

# do not buy anthing on day one
# then
# choose x laggers with lowest out-degree
# then just follow those relationships

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
    stock_names[0] = "A"
    n = len(stock_names)
    total_pairs = n * n
    typed_stock_names = List(stock_names)
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
            result = process_pair(typed_dict, n, typed_stock_names, total_pairs)
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
