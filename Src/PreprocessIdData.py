import torch
import platform
#import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
import copy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import os

transactions_df = pd.read_csv('Data/Raw/transactions_train.csv')
product = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))


# Preprocess customer dataframe
# check for NaN and make a subset with relevant columns
customers
percent_c = (customers.isnull().sum()/customers.isnull().count()*100).sort_values(ascending = False)

# we cannot use FN and Active. They have a lot of NaN values. We keep the rest
customers = customers[['customer_id','age', 'club_member_status', 'postal_code', 'fashion_news_frequency']]
customers = customers.dropna(subset=['age'])




# Preprocess article dataframe
product
# From the articles we see several columns with the same information. section_name, product_group_name and garment_group_name gives almost the same information just with differnet headlines.
# we can discard the garment_group_name and product_group_name and the number associated. 
# we can also discard all the colour codes and only keep the colour_group_name.
# The detailed desciption is also discarded.
# Difference between product_code and product_type_no is product_code gives the number for the same item, while product_type_no is the number for all the trousers fx. 

# subset relevant columns
product = product[['article_id','product_code','product_type_no', 'prod_name', 'product_type_name', 'colour_group_name', 'department_name', 'section_name']]
# check for NaN
percent_a = (product.isnull().sum()/product.isnull().count()*100).sort_values(ascending = False)




# Preprocess transaction train dataframe
#datetime and create a month column
transactions_df
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df = transactions_df.dropna(subset=['price'])

# Add season column
transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 
# drop month column
transactions_df = transactions_df.drop("month", axis=1)



# Merge dataframes into an enriched dataframe with all features
transactions_data = transactions_df.merge(product[['article_id']], how = 'left', on = 'article_id')
transactions_data_enriched = transactions_data
transactions_data_enriched = transactions_data_enriched.merge(customers[['customer_id','age']], how = 'left', on = 'customer_id')
transactions_data_enriched = transactions_data_enriched.merge(product[['article_id','colour_group_name','department_name']])

# Make a split to train, valid and test
train = transactions_data.iloc[:splitrange]
valid = transactions_data.iloc[splitrange+1:splitrange2]
test = transactions_data.iloc[splitrange2:]

train_enriched = transactions_data_enriched.iloc[:splitrange].drop(['t_dat'], axis = 1)
valid_enriched = transactions_data_enriched.iloc[splitrange+1:splitrange2].drop(['t_dat'], axis = 1)
test_enriched = transactions_data_enriched.iloc[splitrange2:].drop(['t_dat'], axis = 1)

""" 
train_sub = train[['customer_id','article_id']]
valid_sub = valid[['customer_id','article_id']]
test_sub = test[['customer_id','article_id']]

train_sub.to_csv('Data/Preprocessed/TrainData.csv',index=False)
valid_sub.to_csv('Data/Preprocessed/ValidData.csv',index=False)
test_sub.to_csv('Data/Preprocessed/TestData.csv',index=False)
 """

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

num_colours = product['colour_group_name'].nunique()
num_departments = product['department_name'].nunique()
num_articles = product['article_id'].nunique()
Article_id_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_articles+1).fit(product[['article_id']].to_numpy().reshape(-1, 1))
Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(product[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(product[['department_name']].to_numpy().reshape(-1, 1))

Season_encoder = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=5).fit(transactions_data_enriched[['season']])

product_aggregated = train_enriched[['price','article_id','age','sales_channel_id']].groupby(['article_id']).mean().reset_index()
product_aggregated = product[['article_id','prod_name','colour_group_name','department_name']].merge(product_aggregated, how = 'left', on = 'article_id')
product_aggregated['colour_group_name'] = Colour_Encoder.transform(product_aggregated[['colour_group_name']].to_numpy().reshape(-1, 1))
product_aggregated['department_name'] = Department_encoder.transform(product_aggregated[['department_name']].to_numpy().reshape(-1, 1))
product_aggregated['article_id'] = Article_id_encoder.transform(product_aggregated[['article_id']].to_numpy().reshape(-1, 1))

product_aggregated = product_aggregated.groupby(['article_id']).mean().reset_index()
#product_aggregated = product_aggregated.reset_index()
#product_aggregated = product_aggregated.merge(product[['prod_name','colour_group_name','department_name']], how = 'left', on = 'prod_name')

customer_aggregated = colour_df_test.merge(department_df_test, how = 'left', on =  'customer_id', suffixes = ('_colour','_department'))
customer_aggregated = customer_aggregated.merge(customers[['customer_id','age']], how = 'left', on = 'customer_id')

customer_aggregated_extra = train[['customer_id','price','sales_channel_id']].groupby('customer_id').mean()
customer_aggregated = customer_aggregated.merge(customer_aggregated_extra, how = 'left', on = 'customer_id')





#######################



Customer_data = customer_aggregated
Product_data = product_aggregated
""" 
customers = pd.read_csv('Data/Raw/customers.csv')
product = pd.read_csv('Data/Raw/articles.csv') """

num_customers = customers['customer_id'].nunique()
num_products = product['article_id'].nunique()
num_departments = product['department_name'].nunique()
num_colours = product['colour_group_name'].nunique()
max_age = 110


Customer_id_Encoder = preprocessing.LabelEncoder().fit(customers['customer_id'])
Product_Encoder = preprocessing.LabelEncoder().fit(product['article_id'])
Customer_data['customer_id'] = Customer_id_Encoder.transform(Customer_data['customer_id'])
#Product_data['article_id'] = Product_Encoder.transform(Product_data['article_id'])
#Product_data = Product_data.drop(columns=['prod_name'], axis = 1)

Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(product[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(product[['department_name']].to_numpy().reshape(-1, 1))
Age_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=110).fit(customers[['age']].to_numpy().reshape(-1, 1))

all_prices = np.concatenate((Product_data['price'].unique(),Customer_data['price'].unique()),axis = 0)
all_prices = np.unique(all_prices)
num_prices = len(all_prices)



Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(all_prices.reshape(-1, 1))


train_dataset = copy.deepcopy(train_enriched)
train_dataset['price'] = train_dataset['price'].round(decimals=4)
train_dataset.customer_id = Customer_id_Encoder.transform(train_enriched.customer_id.values)
train_dataset['product_id'] = Product_Encoder.transform(train_enriched.article_id.values)
train_dataset['colour_group_name'] = Colour_Encoder.transform(train_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
train_dataset['department_name'] = Department_encoder.transform(train_dataset[['department_name']].to_numpy().reshape(-1, 1))
train_dataset['age'] = Age_encoder.transform(train_dataset[['age']].to_numpy().reshape(-1,1))
train_dataset['price'] = Price_encoder.transform(train_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)




valid_dataset = copy.deepcopy(valid_enriched)
valid_dataset.customer_id = Customer_id_Encoder.transform(valid_enriched.customer_id.values)
valid_dataset['product_id'] = Product_Encoder.transform(valid_enriched.article_id.values)
valid_dataset['colour_group_name'] = Colour_Encoder.transform(valid_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
valid_dataset['department_name'] = Department_encoder.transform(valid_dataset[['department_name']].to_numpy().reshape(-1, 1))
valid_dataset['age'] = Age_encoder.transform(valid_dataset[['age']].to_numpy().reshape(-1,1))
valid_dataset['price'] = Price_encoder.transform(valid_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)


test_dataset = copy.deepcopy(test_enriched)
test_dataset.customer_id = Customer_id_Encoder.transform(test_enriched.customer_id.values)
test_dataset['product_id'] = Product_Encoder.transform(test_enriched.article_id.values)
test_dataset['colour_group_name'] = Colour_Encoder.transform(test_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
test_dataset['department_name'] = Department_encoder.transform(test_dataset[['department_name']].to_numpy().reshape(-1, 1))
test_dataset['age'] = Age_encoder.transform(test_dataset[['age']].to_numpy().reshape(-1,1))
test_dataset['price'] = Price_encoder.transform(test_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)


# Save the traning, validation and test dataset
train_enriched.to_csv('Data/Preprocessed/TrainData_enriched.csv',index=False)
valid_enriched.to_csv('Data/Preprocessed/ValidData_enriched.csv',index=False)
test_enriched.to_csv('Data/Preprocessed/TestData_enriched.csv',index=False)


# Save a subset for improving model performance in training
train_df_sub, valid_df_sub = train_enriched.iloc[0:100000], valid_enriched[0:100000]

train_df_sub.to_csv('Data/Preprocessed/TrainData_enriched_sub.csv',index=False)
valid_enriched.to_csv('Data/Preprocessed/ValidData_enriched_sub.csv',index=False)




#test_dataset.to_csv('Data/Preprocessed/test_final.csv',index=False)

#Product_data['colour_group_name'] = Colour_Encoder.transform(Product_data[['colour_group_name']].to_numpy().reshape(-1, 1))
#Product_data['department_name'] = Department_encoder.transform(Product_data[['department_name']].to_numpy().reshape(-1, 1))
#Product_data = Product_data[['article_id','age','price','sales_channel_id','colour_group_name','department_name']]
Product_data['age'] = Product_data['age'].round(decimals = 0)
Product_data['age'] = Age_encoder.transform(Product_data[['age']].to_numpy().reshape(-1,1))
Product_data['price'] = Product_data['price'].round(decimals=4)
Product_data['price'] = Price_encoder.transform(Product_data[['price']].to_numpy().reshape(-1,1))

train_dataset = train_dataset[['customer_id','product_id','age','price','colour_group_name','department_name']]

Customer_data['colour_group_name'] = Customer_data['max_colour'].str.replace('colour_group_name_','')
Customer_data['department_name'] = Customer_data['max_department'].str.replace('department_name_','')
Customer_data = Customer_data.drop(columns=['max_colour','max_department'], axis = 1)

Customer_data['colour_group_name'] = Colour_Encoder.transform(Customer_data[['colour_group_name']].to_numpy().reshape(-1, 1))
Customer_data['department_name'] = Department_encoder.transform(Customer_data[['department_name']].to_numpy().reshape(-1, 1))
Customer_data['price'] = Customer_data['price'].round(decimals=4)
Customer_data['price'] = Price_encoder.transform(Customer_data[['price']].to_numpy().reshape(-1,1))



Customer_data.to_csv('Data/Preprocessed/Customers_enriched.csv',index = False)
Product_data.to_csv('Data/Preprocessed/Products_enriched.csv',index = False)


