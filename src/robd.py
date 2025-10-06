import numpy as np

def robd_single_sequence(y_norm, m=5, la1=1, la2=0):
    """
    Run R-OBD for a single sequence, return action sequence x.

    y_norm: numpy array, range: [0,1]
    """
    x = np.zeros_like(y_norm, dtype=float)
    x[0] = y_norm[0]
    denom = m + 2 * la1 + la2
    for t in range(1, len(y_norm)):
        x[t] =(m * y_norm[t] + 2 * la1 * x[t - 1]) / denom
    return x

def compute_cost(x_seq, y_norm, m=5.0):
    hitting = (m/2.0) * np.sum((x_seq - y_norm) ** 2)

    if len(x_seq) > 1:
        switching = 0.5 * np.sum((x_seq[1:] - x_seq[:-1]) ** 2)

    else:
        switching = 0.0
    return hitting + switching
