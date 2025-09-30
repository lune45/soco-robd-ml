import os
import numpy as np
import matplotlib.pyplot as plt
import math
from src.preprocess import load_normalize, sequences
from src.robd import robd_single_sequence, compute_cost

# import dataset
data_path = os.path.join("data", "AI_workload.csv")

# load and normolize
y_norm, y_min, y_max = load_normalize(data_path)
print("loaded samples:", len(y_norm))

# cut time sequence
window_size = 24
step = 2
sequence = sequences(y_norm, window_size = window_size, step = step)
print("Number of sequences:", sequence.shape[0])

# R-OBD parameters
m = 5.0
la1 = 2.0 / (1.0 + math.sqrt(1.0 + (4.0 * m * m) / (m * m))) #2.0 / (1.0 + math.sqrt(1.0 + (4.0 * beta**2) / (alpha * m)))

la2 = 0

# run R-OBD
costs = []
for seq in sequence:
    x = robd_single_sequence(seq, m = m, la1 = la1, la2 = la2)
    c = compute_cost(x, seq, m = m)
    costs.append(c)
costs = np.array(costs)
print("Average cost:", costs.mean())
print("Median cost", np.median(costs))

# save costs
os.makedirs("results", exist_ok = True)
np.savetxt("results/robd_cost_1.txt", costs)