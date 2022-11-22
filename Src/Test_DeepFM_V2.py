import torch
import pandas as pd
from Src.Create_recommendations import Get_Recommendations
import numpy as np
from Src.deepFM import DeepFactorizationMachineModel

test_df_negatives = pd.read_csv('Data/Preprocessed/test_with_negative_subset.csv')
test_df = pd.read_csv('Data/Preprocessed/test_df_subset_subset.csv')
#test_df = test_df_negatives[test_df_negatives['target']==1]
customers = test_df.customer_id.unique()
customers_2 = test_df_negatives.customer_id.unique()

A = set(customers)
B = set(customers_2)

differenceA = A-B
differnceB = B-A
difference = differenceA|differnceB

one_accuracy_all = []
twelve_accuracy_all = []
batch_size = 1024
model = torch.load('Models/DeepFM_model.pth')
num_recommendations = 12
for c in customers:
    temp_accuracy = []
    items, true_values = Get_Recommendations(c, model, test_df, test_df_negatives, batch_size=batch_size, num_recommendations = num_recommendations)
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

