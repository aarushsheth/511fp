from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import numba
from numba import njit
from numba.typed import Dict
from numba.typed import List
import networkx as nx
import pickle
@njit
#################################################################
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
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    results = []
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        filtered_data = stockData[stockData["datadate"] == datetime.strptime(
            date_str, "%Y-%m-%d")]
        if not filtered_data.empty:
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
def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=8, font_weight='bold', node_color='orange', alpha=0.8, arrowsize=12)
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
#################################################################
# I. Processing lead-lag effect
stockData = pd.read_csv("finally.csv")
stockData['datadate'] = pd.to_datetime(stockData['datadate'])
stockData = stockData.sort_values(by='datadate')

with open("newstocks.txt", "r") as f:
    x = f.read().splitlines()
x[0] = "A"
n = len(x)
total_pairs = n * n
typed_x = List(x)
start_date = '2013-03-15'
end_date = '2023-03-15'

month_data = process_month(start_date, end_date)
with open('month_data.pkl', 'wb') as f:
    pickle.dump(month_data, f)
#################################################################
# II. Performing network analysis
# with open('month_data.pkl', 'rb') as f:
#     loaded_month_data = pickle.load(f)

# for adj_matrix in month_data:
#     G = build_graph(adj_matrix, x)
    # Uncomment to run the plots and perform calcs
    # visualize_graph(G) 
    # print("Modularity:", modularity(G))
    # print("Eigenvector Centralities:", eigenvector_centrality(G))
    # print("Diameter:", diameter(G))
    # print("Average Path Length:", average_path_length(G))
    # print("Average Node Degree:", average_node_degree(G))
    # print("Clustering Coefficients:", clustering_coefficients(G))
    # print("\n")
#################################################################
# III. Constructing portfolio
#need to look at most correlated, wait for time when leader falls/rises 
#which is when we short/long the lagger
#4 stocks in the portfoli around there
