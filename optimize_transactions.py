"""
An optimization model to optimize the number of transactions to be made between a set of people, when the
net transaction for each person is given.

The data should be a .csv on the following format:

    Name, Net_Transaction
    Person1, 1000
    Person2: -500
    Person3: -500

Net_Transactions must  be positive if the person should receive money, and negative if the person should pay money.
The sum of Net_Transactions must be 0. One solution to the problem above could be:

    Person 2 transfers 500 to Person 1
    Person 3 transfers 500 to Person 1

Author: Peter Sandberg
"""

import csv
from pulp import *


CURRENCY = "kr"
SLACK = "_slack"


with open("data.csv") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Header
    data = [row for row in csv_reader]

net_transactions = {name: float(trans) for name, trans in data}
names = list(net_transactions.keys())

# ----- INDEX -----
name_combos = [(n1, n2) for n1 in names for n2 in names if n1 != n2]

# ----- VARIABLES -----
transfer = LpVariable.dicts("Transfer", name_combos, 0, None)
slack = LpVariable.dicts("Slack", names, -1.0, 1.0)

# ----- PROBLEM -----
prob = LpProblem("Transaction optimization", LpMinimize)

# ----- OBJECTIVE -----
prob += lpSum([transfer[nc] for nc in name_combos]), "Transaction_sum"


for n1 in names:
    prob += lpSum([transfer[(n2, n1)] for n2 in names if n1 != n2]) - lpSum([transfer[(n1, n2)] for n2 in names if n1 != n2]) + slack[n1] == net_transactions[n1], f"Net_{n1}"

# Write problem to file
prob.writeLP("TransactionOptimization.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print(f"Status: {prob.status}")


actual_net_transactions = {}

max_name_len = max(len(n) for n in names)

print("Transfers:")
for n1 in names:
    actual_net_transactions[n1] = 0
    for n2 in names:
        if n1 == n2:
            continue
        t_out = transfer[(n1, n2)].varValue
        if t_out != 0:
            print(f"{n1:<{max_name_len+1}} -> {n2:<{max_name_len+1}} {t_out} {CURRENCY}")
        actual_net_transactions[n1] += transfer[(n2, n1)].varValue - t_out

print("-" * 20)
print("Net transactions:")
for n in names:
    print(f"{n:<{max_name_len+1}} {actual_net_transactions[n]} {CURRENCY} (Expected: {net_transactions[n]} {CURRENCY})")
