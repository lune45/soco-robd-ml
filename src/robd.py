import numpy as np

def robd_single_sequence(y_seq, m=5, la1=1, la2=0):
    """
    run R-OBD for single sequence, return sequence x

    y_seq: numpy array, range:[0,1]
    """
    x = np.zero_like(y_seq, dtype=float)
    x[0] = y_seq[0]
    denom = m + 2 * la1 + la2
    for t in range(1, len(y_seq)):
        x[t] =(m * y_seq[t] + 2 * la1 * x[t - 1]) / denom
    return x

def compute_cost(x_seq, y_seq, m=5.0):
    hitting = (m/2.0) * np.sum((x_seq - y_seq) ** 2)

    if len(x_seq) > 1:
        switching = 0.5 * np.sum((x_seq[1:] - x_seq[:-1]) ** 2)

    else:
        switching = 0.0
    return hitting + switching
