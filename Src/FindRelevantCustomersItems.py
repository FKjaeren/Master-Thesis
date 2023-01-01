import pandas as pd

#script for filtering and only get the relevant data

# Read data
df = pd.read_csv('Data/Raw/transactions_train.csv')


#Get different stats for each item
item_agg = df.groupby('article_id').aggregate({'article_id':'count','t_dat':'max'})

#Get different stats for each customer
cust_agg = df.groupby('customer_id').aggregate({'customer_id':'count','t_dat':'max'})

# Get subsets of the just above found stats.
item_agg_subset = item_agg[item_agg['t_dat']>'2020-03-22']

cust_agg_subset = cust_agg[cust_agg['t_dat']>'2020-03-22']

## Get an even smaller subset.

cust_agg_subset_v2 = cust_agg_subset[(cust_agg_subset['t_dat']>'2020-07-22')]
cust_agg_subset_v2 = cust_agg_subset_v2[cust_agg_subset_v2['customer_id']>5]
cust_agg_subset_v2 = cust_agg_subset_v2.rename(columns = {'customer_id':'purchase_count'}).reset_index()
item_agg_subset2 = item_agg[item_agg['article_id']>=15]
item_agg_subset2 = item_agg_subset2.rename(columns = {'article_id':'purchase_count'}).reset_index()

transaction_subset = df[df['customer_id'].isin(cust_agg_subset_v2.customer_id)]
transaction_subset = transaction_subset[transaction_subset['article_id'].isin(item_agg_subset2.article_id)]

transaction_subset = transaction_subset[transaction_subset['t_dat']>'2019-09-22']

transaction_subset.to_csv('Data/Raw/transactions_train_subset.csv', index = False)

articles = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')

articles_subset = articles[articles['article_id'].isin(transaction_subset.article_id)]
customers_subset = customers[customers['customer_id'].isin(transaction_subset.customer_id)]

## Save these subsets in the raw datafodler as this data will now be used. Argumentation for this can be found in the thesis.s

articles_subset.to_csv('Data/Raw/articles_subset.csv', index = False)
customers_subset.to_csv('Data/Raw/customers_subset.csv', index = False)
