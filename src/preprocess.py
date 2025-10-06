import pandas as pd
import numpy as np

def load_normalize(csv_path):
    """
    Read data from CSV.

    Normalize data to [0,1].

    Return normalized sequence, global min, global max.
    """
    df = pd.read_csv(csv_path) # read csv to pandas
    y = df.iloc[:, 1].values.astype(float) # read Power
    y_min, y_max = y.min(), y.max() #record min/max
    if y_max == y_min:
        y_norm = np.zeros_like(y, dtype=float)
    else:
        y_norm = (y - y_min) / (y_max - y_min) #normalize to [0,1]
    return y_norm, y_min, y_max

def sequences(y_norm, window_size=24, step=2):
    """
    Use a sliding window to cut a long sequence into short sequences.

    Return numpy array, shape = (num_sequences, window_size)
    """
    sequences = []
    # start from 0 to len(y_norm)-window_size, move by 'step' each time
    for start in range(0, len(y_norm) - window_size +1, step):
        sequences.append(y_norm[start: start + window_size])
    return np.array(sequences)