"""
An optimization model to optimize the number of transactions to be made between a set of people, when the
net transaction for each person is given.

The data should be a .csv on the following format:

    Name, Net_Transaction
    Person1, 1000
    Person2: -500
    Person3: -500

The sum of Net_Transactions must be 0

Author: Peter Sandberg
"""

import csv
from pulp import *

with open("data.csv") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Header
    data = [row for row in csv_reader]

net_transactions = {name: float(trans) for name, trans in data}
names = list(net_transactions.keys())

# ----- INDEX -----
name_combos = [(n1, n2) for n1 in names for n2 in names if n1 != n2]
# Unique name combos, i.e. contains only one of [(Person1, Person2), (Person2, Person1)]
name_combos_u = [(n1, n2) for i, n1 in enumerate(names) for n2 in names[i+1:]]

# ----- PARAMETERS -----
max_payout = max(net_transactions.values())
max_payment = abs(min(net_transactions.values()))

# ----- VARIABLES -----
payout = LpVariable.dicts("Payout", name_combos, 0, None)
payment = LpVariable.dicts("Payment", name_combos, None, 0)
is_trans = LpVariable.dicts("Is_Transaction", name_combos_u, 0, 1, LpInteger)

# ----- PROBLEM -----
prob = LpProblem("Transaction optimization", LpMinimize)

# ----- OBJECTIVE -----
prob += lpSum([is_trans[nc] for nc in name_combos_u]), "Num_Transactions"

# ----- CONSTRAINTS -----
for n1, n2 in name_combos:
    # The amount Person1 receives from Person2 must equal the amount Person2 transferred to Person1
    prob += payout[(n1, n2)] + payment[(n2, n1)] == 0, f"Net_is_zero_{(n1, n2)}"

for i, n1 in enumerate(names):
    for n2 in names[i+1:]:
        # If the payout is positive, is_trans must be 1
        prob += payout[(n1, n2)] - max_payout * is_trans[(n1, n2)] <= 0, f"Is_payout_{(n1, n2)}"
        prob += payout[(n2, n1)] - max_payout * is_trans[(n1, n2)] <= 0, f"Is_payout_{(n2, n1)}"

        # If the payout is positive, is_trans must be 1
        prob += payment[(n1, n2)] + max_payment * is_trans[(n1, n2)] >= 0, f"Is_payment_{(n1, n2)}"
        prob += payment[(n2, n1)] + max_payment * is_trans[(n1, n2)] >= 0, f"Is_payment_{(n2, n1)}"

for n1 in names:
    if net_transactions[n1] >= 0:
        prob += lpSum([payout[(n1, n2)] for n2 in names if n1 != n2]) == net_transactions[n1], f"Payout_{n1}"
        prob += lpSum([payment[(n1, n2)] for n2 in names if n1 != n2]) == 0, f"Payment_{n1}"
    else:
        prob += lpSum([payout[(n1, n2)] for n2 in names if n1 != n2]) == 0, f"Payout_{n1}"
        prob += lpSum([payment[(n1, n2)] for n2 in names if n1 != n2]) == net_transactions[n1], f"Payment_{n1}"

# Write problem to file
prob.writeLP("TransactionOptimization.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print(f"Status: {prob.status}")

# Each of the variables is printed with it's resolved optimum value
print(f"Number of transactions: {value(prob.objective)}")

# Print payments
print("Payments")
for v in prob.variables():
    if "Payment" in v.name and v.varValue != 0:
        print(f"{v.name} = {v.varValue}")


