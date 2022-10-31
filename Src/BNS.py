import numpy as np
import pandas as pd
import torch

def BNS(i, negative_items, y, prior, size, alpha):
    def sigmoid(x):
        if x > 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))
    x_ui = y[i.numpy()]
    negative_scores = y[negative_items.numpy()]
    length = len(negative_scores) + 1
    candidate_set = np.random.choice(negative_items, size=size, replace=False)  #O(|I|)
    candidate_scores = [y[int(l)] for l in candidate_set]
    # step 1 : computing info(l)
    info = np.array([1 - sigmoid((x_i - x_ul).detach())  for x_ul in candidate_scores for x_i in x_ui])                #O(1)
    # step 2 : computing prior probability
    p_fn = np.array([prior[int(l)] for l in candidate_set ])                                 #O(1)
    # step 3 : computing empirical distribution function (likelihood)
    F_n = np.array([(sum(negative_scores <= x_ul) / length).item() for x_ul in candidate_scores])   #O(|I|)
    # step 4: computing posterior probability
    unbias = (1 - F_n) * (1 - p_fn) / (1 - F_n - p_fn + 2 * F_n * p_fn)                      #O(1)
    # step 5: computing conditional sampling risk
    conditional_risk = (1-unbias) * info - alpha * unbias * info                             #O(1)
    j = candidate_set[conditional_risk.argsort()[0]]
    return j