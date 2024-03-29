import numpy as np
import torch
import pandas as pd
# The runtime of all test

#DeepFM
value_1 = 33157.21/12000
value_2 = 36477.17/12000
value_3 = 32837.07/12000

#FM 
value_2 = 10345.90/12000
value_1 = 10287.15/12000
value_3 = 0.7868592017690341
value_4 = 0.7745043141841889

#Baseline
value_1 = 457.65/12000
value_2 = 440.30/12000
value_3 = 456.94/12000

# MLP
value_1 = 14459.32/12000
value_2 = 14277.88/12000
value_3 = 1.1367792414228122
value_4 = 1.1522578936616579

# MF multi dim
value_1 = 1016.06/12000
value_2 = 1072.99/12000
value_3 = 0.08169
value_4 = 0.08002

value_1 = 487.480064609468/12000

#Use model for prediction time:
value_1 = 86.53
value_2 = 85.86

## Use model for prediciton time Local:
value_1 = 19.17
value_2 = 20.19

#values = [value_1,value_2,value_3]
values = [value_3, value_4]
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