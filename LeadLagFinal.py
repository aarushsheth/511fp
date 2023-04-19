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
def find_all_leader_lagger_pairs(masked_array, stock_names):
    leader_lagger_pairs = {}
    for row in range(masked_array.shape[0]):
        for col in range(masked_array.shape[1]):
            if row != col and row < len(stock_names) and col < len(stock_names):
                leader = stock_names[row]
                lagger = stock_names[col]
                leader_lagger_pairs[(leader, lagger)] = masked_array[row, col]
    return leader_lagger_pairs
def create_directed_graph(leader_lagger_dict, stock_colors):
    G = nx.DiGraph()
    for leader, lagger in leader_lagger_dict.items():
        G.add_edge(lagger, leader)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 10))
    for stock, color in stock_colors.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[stock], node_color=color, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels={stock: stock}, font_size=12, font_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=20, node_size=1000, alpha=0.5, width=2)
    plt.axis('off')
    plt.show()

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
# Use for bull market
# end_date = stockData.index[-251]
# start_date = end_date - pd.Timedelta(days=249)
# filtered_stock_data = stockData.loc[start_date:end_date, stocks_to_include]
filtered_stock_data = stockData.loc[:, stocks_to_include].tail(250)
n_laggers = len(leader_lagger_dict.values())
initial_investment = 500000
remaining_cash = initial_investment
investment_per_stock = initial_investment / n_laggers
n_days = filtered_stock_data.shape[0]
portfolio = {stock: 0 for stock in leader_lagger_dict.values()}

trailing_stop_percentage = 0.10
highest_leader_prices = {leader: 0 for leader in leader_lagger_dict.keys()}
remaining_cash = initial_investment
portfolio_cash = {stock: 0 for stock in leader_lagger_dict.values()}

for i in range(1, n_days):
    for leader, lagger in leader_lagger_dict.items():
        current_leader_price = filtered_stock_data[leader].iloc[i]
        highest_leader_prices[leader] = max(highest_leader_prices[leader], current_leader_price)
#         Use for bull market
#         if current_leader_price >= 1.0 * filtered_stock_data[leader].iloc[i - 1]:
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
            portfolio

rounded_portfolio = {k: round(v, 2) for k, v in portfolio.items()}
print(rounded_portfolio)
initial_value = initial_investment
final_value = portfolio_value(portfolio, filtered_stock_data.iloc[-1]) + remaining_cash
portfolio_return = (final_value - initial_value) / initial_value
print(f"Portfolio return over the period: {portfolio_return * 100:.2f}%")
stock_prices_list = filtered_stock_data.values.T
portfolio_values = [portfolio_value(portfolio, filtered_stock_data.iloc[i]) + remaining_cash for i in range(n_days)]

ticker = "^GSPC"
stock = yf.Ticker(ticker)
# Use for bull market
# historical_data = stock.history(start="2021-06-15", end="2022-04-15")
historical_data = stock.history(start="2022-03-29", end="2023-03-15")
historical_returns = historical_data['Open'].pct_change().dropna()
resampled_historical_returns = np.interp(
    np.linspace(0, len(historical_returns) - 1, num=len(portfolio_values)),
    np.arange(len(historical_returns)),
    historical_returns
)
cumulative_historical_returns = (resampled_historical_returns + 1).cumprod() - 1
cumulative_portfolio_returns = [(value - initial_value) / initial_value for value in portfolio_values]
percentage_historical_returns = 100 * cumulative_historical_returns
percentage_portfolio_returns = 100 * np.array(cumulative_portfolio_returns)
percentage_portfolio_returns -= percentage_portfolio_returns[0]

tick_locations = np.linspace(0, len(portfolio_values) - 1, num=5, dtype=int)
tick_labels = [filtered_stock_data.index[i].strftime("%b %Y") for i in tick_locations]
plt.plot(percentage_historical_returns, label="S&P 500 Returns")
plt.plot(percentage_portfolio_returns, label="Portfolio Returns")
plt.ylabel("Returns (%)")
# Use for bull market
# plt.title("Trailing Stop: 0.10 and Buy Threshold: 1.0")
plt.title("Trailing Stop: 0.10 and Buy Threshold: 1.075")
plt.legend()
plt.xticks(tick_locations, tick_labels)
plt.show()

stock_colors = {stock: 'red' for stock in leader_lagger_dict.keys()}
stock_colors.update({stock: 'blue' for stock in leader_lagger_dict.values()})
create_directed_graph(leader_lagger_dict, stock_colors)
first_day_data = loaded_daily_data[0, :, :]
masked_array_first_day = np.ma.masked_array(first_day_data, mask=np.eye(498, dtype=bool))
first_day_leader_lagger_pairs = find_all_leader_lagger_pairs(masked_array_first_day, stock_names)
filtered_first_day_leader_lagger_pairs = {
    (leader, lagger): value
    for (leader, lagger), value in first_day_leader_lagger_pairs.items()
    if leader in stockData.columns and lagger in stockData.columns
}
first_day_stock_colors = {}
for leader, lagger in filtered_first_day_leader_lagger_pairs.keys():
    first_day_stock_colors[leader] = 'red'
    first_day_stock_colors[lagger] = 'blue'
create_directed_graph(filtered_first_day_leader_lagger_pairs, first_day_stock_colors)
