import numpy as np

def robd_single_sequence(y_seq, m=5, la1=1, la2=0):
    """
    run R-OBD for single sequence, return sequence x

    y_seq: numpy array, range:[0,1]
    """
    x = np.zero_like(y_seq, dtype=float)
    x[0] = y_seq[0]
    denom = m + 2 * la1 +la2
    for t in range(1, len(y_seq)):
        x[t] =(m * y_seq[t] + 2* la1 *x[t - 1])/ denom
    return x
