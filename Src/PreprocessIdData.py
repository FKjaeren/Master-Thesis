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
transactions_data_enriched = transactions_data_enriched.merge(product[['article_id','colour_group_name','department_name']])

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

colours = product['colour_group_name'].unique()
departments = product['department_name'].unique()
customer_ids = customers['customer_id']

colour_df = pd.get_dummies(train_enriched[['customer_id','colour_group_name']],columns = ['colour_group_name'])
colour_df_test = colour_df.groupby('customer_id').sum()
colour_df_test['max'] = colour_df_test.idxmax(axis=1)
colour_df_test = colour_df_test.reset_index()[['customer_id','max']]

department_df = pd.get_dummies(train_enriched[['customer_id','department_name']],columns = ['department_name'])
department_df_test = department_df.groupby('customer_id').sum()
department_df_test['max'] = department_df_test.idxmax(axis=1)
department_df_test = department_df_test.reset_index()[['customer_id','max']]


product_aggregated = train_enriched[['price','prod_name','age','sales_channel_id']].groupby(['prod_name']).mean()
product_aggregated = product_aggregated.reset_index()
product_aggregated = product_aggregated.merge(product[['prod_name','colour_group_name','department_name']], how = 'left', on = 'prod_name')

customer_aggregated = colour_df_test.merge(department_df_test, how = 'left', on =  'customer_id', suffixes = ('_colour','_department'))
customer_aggregated = customer_aggregated.merge(customers[['customer_id','age']], how = 'left', on = 'customer_id')

customer_aggregated.to_csv('Data/Preprocessed/Customers_enriched.csv',index = False)
product_aggregated.to_csv('Data/Preprocessed/Products_enriched.csv',index = False)