import numpy as np
import json


with np.load("function_test.npz", allow_pickle = True) as f:
    GRID = f["GRID"]
    print(f)
    if 'GRID' not in f: raise ValueError("File does not contain a valid GRID")
