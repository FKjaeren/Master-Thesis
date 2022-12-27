import torch
from Src.deepFM import CreateDataset, DatasetIter
import math

def Get_Recommendations(customer_id, model, test_set, test_full_set, num_recommendations = 6):
    true_values = test_set.article_id.unique()
    test_tensor = torch.tensor(test_full_set.fillna(0).to_numpy(), dtype = torch.int)
    del test_full_set
    outputs, _ = model(test_tensor)
    prob, idx = torch.topk(outputs, num_recommendations)
    items = test_tensor[idx,1]
    return items, true_values.astype(int)
