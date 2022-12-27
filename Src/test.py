import numpy as np
import torch
import pandas as pd
items = torch.tensor([1,10,12,2,3,5,8])
true_values = [1,4,9,11,29,33,75,9,13,8]
num_recommendations = 6

twelve_accuracy_all = []
temp_accuracy = []

temp_accuracy = []
for i in items:
    if i in true_values:
        temp_accuracy.append(1)
        print("hej")
    else:
        temp_accuracy.append(0)
if(num_recommendations <= len(true_values)):
    temp_accuracy_final = sum(temp_accuracy)/num_recommendations
else:
    temp_accuracy_final = sum((np.sort(temp_accuracy)[0:len(true_values)]))/len(true_values)

twelve_accuracy_all.append(temp_accuracy_final)

#DeepFM
value_1 = 33157.21/12000
value_2 = 36477.17/12000

#FM 
value_2 = 10345.90/12000
value_1 = 10287.15/12000

#Baseline
value_1 = 457.65/12000
value_2 = 440.30/12000
value_3 = 456.94/12000

# MLP
value_1 = 14459.32/12000
value_2 = 14277.88/12000

#values = [value_1,value_2,value_3]
values = [value_1, value_2]
np.mean(values)
np.std(values)

#Get average amount of purchases in the test dataset.

test_df = pd.read_csv('Data/Preprocessed/test_df_subset.csv')
customer_purchases_df = test_df[['customer_id','article_id']].groupby('customer_id').count()
print(f"The average amount of purhcase per customer is: {customer_purchases_df.mean()}")
print(f"The min amount of purhcase per customer is: {customer_purchases_df.min()}")
print(f"The max amount of purhcase per customer is: {customer_purchases_df.max()}")
print(f"The median amount of purhcase per customer is: {customer_purchases_df.median()}")

## Calculate std for data load runningtime experiment.
#Local
value_1_local = 11.441784143447876
value_2_local = 11.531735897064209
values_local = [value_1_local, value_2_local]
np.mean(values_local)
np.std(values_local)

#Cloud:
value_1 = 62.59980630874634
value_2 = 64.00353407859802
values = [value_1, value_2]
np.mean(values)
np.std(values)