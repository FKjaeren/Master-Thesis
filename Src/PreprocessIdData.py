import numpy as np
import pandas as pd
import torch
transactions_df = pd.read_csv('Data/Raw/transactions_train.csv')
splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]


train_sub = train[['customer_id','article_id']]
valid_sub = valid[['customer_id','article_id']]
test_sub = test[['customer_id','article_id']]

train_sub.to_csv('Data/Preprocessed/TrainData.csv',index=False)
valid_sub.to_csv('Data/Preprocessed/ValidData.csv',index=False)
test_sub.to_csv('Data/Preprocessed/TestData.csv',index=False)

articles = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
articles_sub = articles[['article_id']].values.flatten()
customers_sub = customers[['customer_id']].values.flatten()