import torch
import pandas as pd
from Src.Create_recommendations import Get_Recommendations
import numpy as np
import pickle
from Src.deepFM import DeepFactorizationMachineModel

test_df_negatives = pd.read_csv('Data/Preprocessed/test_with_negative_subset_part1.csv', nrows = 60000000)
test_df = pd.read_csv('Data/Preprocessed/test_df_subset_final.csv')
#test_df = test_df_negatives[test_df_negatives['target']==1]
customers = test_df.customer_id.unique()

"""
customers = test_df.customer_id.unique()
customers_2 = test_df_negatives.customer_id.unique()

A = set(customers)
B = set(customers_2)

differenceA = A-B
differnceB = B-A
difference = differenceA|differnceB
"""
one_accuracy_all = []
twelve_accuracy_all = []
batch_size = 1024

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)
model = DeepFactorizationMachineModel(field_dims = test_df.columns, embed_dim=26, n_unique_dict = number_uniques_dict, device = 'cpu', batch_size=1,dropout=0.2677)
model.load_state_dict(torch.load('Models/DeepFM_model_Final.pth'))
num_recommendations = 12
chunksize = 200000
test_full_data_path = 'Data/Preprocessed/test_with_negative_subset_part1.csv'
for c in customers:
    temp_accuracy = []
    items, true_values = Get_Recommendations(c, model, test_df, test_full_set=test_df_negatives, test_full_data_path =test_full_data_path ,chunksize = chunksize,batch_size=batch_size, num_recommendations = num_recommendations, iter_data = False)
    if any(x in items for x in true_values):
        accuracy = 1
    else:
        accuracy = 0
    one_accuracy_all.append(accuracy)
    for i in items:
        if i in true_values:
            temp_accuracy.append(1)
        else:
            temp_accuracy.append(0)
    if(num_recommendations <= len(true_values)):
        temp_accuracy = sum(temp_accuracy)/num_recommendations
    else:
        temp_accuracy = sum((np.sort(temp_accuracy)[0:len(true_values)]))/len(true_values)

    twelve_accuracy_all.append(temp_accuracy)

one_accuracy_all = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all = sum(twelve_accuracy_all)/len(customers)

print("The accuracy at hitting one correct recommendation is: ",one_accuracy_all, "%")
print("The accuracy at hitting 12 accurate recommendations is ",twelve_accuracy_all,"%")

