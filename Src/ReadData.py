from posixpath import split
import pandas as pd
import numpy as np

data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')

transactions_df['date'] = pd.to_datetime(transactions_df['t_dat'])

## 80/20 train test split.

splitrange = round(0.8*len(transactions_df['date']))

transactions_train = transactions_df.iloc[:splitrange]
transactions_test = transactions_df.iloc[splitrange+1:]

### Create age intervals on customers:

customer_df['age_interval'] = pd.cut(customer_df['age'],5,right=False)

### Next we will combine the different dataset:

X_train = transactions_train.merge(customer_df,how ='left', on = 'customer_id')
X_train = X_train.merge(articles_df,how = 'left',on = 'article_id')

X_train.to_csv('Data/Preprocessed/X_train.csv',index=False)