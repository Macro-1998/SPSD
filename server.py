import numpy as np


data = np.load('exps/paper/final/gamma10_lambda30_weight1/MSRVTT-test-sims.npy', allow_pickle=True)
for x in data:
    print(x)
