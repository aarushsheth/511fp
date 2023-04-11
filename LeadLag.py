from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import numba
from numba import njit
from numba.typed import Dict
from numba.typed import List


@njit
def process_pair(fuck8, n, x, total_pairs):
    m = np.zeros((498, 498))

    for index in range(total_pairs):
        i = x[index // n]
        j = x[index % n]
        x1 = x.index(i)
        y1 = x.index(j)
        fuck10_values = np.array(
            [fuck8[f"{i}r"], fuck8[f"{j}ryi"], fuck8[f"{j}rys"]])
        if (np.isnan(fuck10_values[0]) or np.isnan(fuck10_values[1]) or np.isnan(fuck10_values[2])):
            continue
        if (fuck10_values[0] >= 0):
            if fuck10_values[0] <= fuck10_values[1] and fuck10_values[0] <= fuck10_values[2]:
                m[x1][y1] = 1

        if (fuck10_values[0] < 0):
            if fuck10_values[0] <= fuck10_values[2] and fuck10_values[0] <= fuck10_values[1]:
                m[x1][y1] = 1
    return m


def process_month(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)

    results = []

    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")

        filtered_data = fuck7[fuck7["datadate"] == datetime.strptime(
            date_str, "%Y-%m-%d")]

        if not filtered_data.empty:
            filtered_data = filtered_data.drop("datadate", axis=1)
            filtered_data = filtered_data.drop("Unnamed: 0", axis=1)
            data_dict = filtered_data.to_dict()

            typed_dict = Dict.empty(
                key_type=numba.types.unicode_type,
                value_type=numba.types.float64
            )

            for key, value in data_dict.items():
                typed_dict[key] = value[next(iter(value))]

            result = process_pair(typed_dict, n, typed_x, total_pairs)
            results.append(result)

    return np.array(results)


# Read CSV file and preprocess the data
fuck7 = pd.read_csv("finally.csv")
fuck7['datadate'] = pd.to_datetime(fuck7['datadate'])
fuck7 = fuck7.sort_values(by='datadate')

# Initialize the matrix and stock symbols list
with open("newstocks.txt", "r") as f:
    x = f.read().splitlines()
x[0] = "A"

# Replace the double for loop with a single loop
n = len(x)
total_pairs = n * n
typed_x = List(x)

start_date = '2023-03-01'
end_date = '2023-03-02'
month_data = process_month(start_date, end_date)
