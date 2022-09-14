import numpy as np
import pandas as pd
transactions_df = pd.read_csv('Data/Raw/transactions_train.csv')
product = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

transactions_data = transactions_df.merge(product[['article_id','prod_name']], how = 'left', on = 'article_id')
transactions_data_enriched = transactions_data
transactions_data_enriched = transactions_data_enriched.merge(customers[['customer_id','age']], how = 'left', on = 'customer_id')
transactions_data_enriched = transactions_data_enriched.merge(product[['article_id','colour_group_code','department_no']])

train = transactions_data.iloc[:splitrange]
valid = transactions_data.iloc[splitrange+1:splitrange2]
test = transactions_data.iloc[splitrange2:]

train_enriched = transactions_data_enriched.iloc[:splitrange].drop(['t_dat','article_id'], axis = 1)
valid_enriched = transactions_data_enriched.iloc[splitrange+1:splitrange2].drop(['t_dat','article_id'], axis = 1)
test_enriched = transactions_data_enriched.iloc[splitrange2:].drop(['t_dat','article_id'], axis = 1)


train_sub = train[['customer_id','prod_name']]
valid_sub = valid[['customer_id','prod_name']]
test_sub = test[['customer_id','prod_name']]

train_sub.to_csv('Data/Preprocessed/TrainData.csv',index=False)
valid_sub.to_csv('Data/Preprocessed/ValidData.csv',index=False)
test_sub.to_csv('Data/Preprocessed/TestData.csv',index=False)

train_enriched.to_csv('Data/Preprocessed/TrainData_enriched.csv',index=False)
valid_enriched.to_csv('Data/Preprocessed/ValidData_enriched.csv',index=False)
test_enriched.to_csv('Data/Preprocessed/TestData_enriched.csv',index=False)
