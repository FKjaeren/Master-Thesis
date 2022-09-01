
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')

# Preprocess customer dataframe
# check for NaN and make a subset with relevant columns
customer_df
percent_c = (customer_df.isnull().sum()/customer_df.isnull().count()*100).sort_values(ascending = False)

# we cannot use FN and Active. They have a lot of NaN values. We keep the rest
df_c = customer_df[['customer_id','age', 'club_member_status', 'postal_code', 'fashion_news_frequency']]
df_c.dropna(subset=['age'])

# Preprocess article dataframe
articles_df
# From the articles we see several columns with the same information. section_name, product_group_name and garment_group_name gives almost the same information just with differnet headlines.
# we can discard the garment_group_name and product_group_name and the number associated. 
# we can also discard all the colour codes and only keep the colour_group_name.
# The detailed desciption is also discarded.
# Difference between product_code and product_type_no is product_code gives the number for the same item, while product_type_no is the number for all the trousers fx. 

# subset relevant columns
df_a = articles_df[['article_id','product_code','product_type_no', 'prod_name', 'product_type_name', 'colour_group_name', 'department_name', 'section_name']]
# check for NaN
percent_a = (df_a.isnull().sum()/df_a.isnull().count()*100).sort_values(ascending = False)

# Preprocess transaction train dataframe
#datetime and create a month column
transactions_df
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month

transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 
# drop month column
transactions_df = transactions_df.drop("month", axis=1)

df_a.to_csv("Data/Preprocessed/df_a.csv")
df_c.to_csv("Data/Preprocessed/df_c.csv")
transactions_df.to_csv("Data/Preprocessed/df_t.csv")






