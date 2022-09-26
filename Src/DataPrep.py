
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import preprocessing

import seaborn as sns
data_path = 'Data/Raw/'

customer_df = pd.read_csv(data_path+'customers.csv')

# Preprocess customer dataframe
# check for NaN and map values in some of the columns 
customer_df
customer_df = customer_df[customer_df.fashion_news_frequency != "None"]

map =  {"NONE":0, "Regularly": 1, "Monthly":2} #np.nan:0
customer_df["fashion_news_frequency"] = customer_df["fashion_news_frequency"].map(map)

map2 =  {np.nan:0, 1.0: 1}
customer_df["FN"] = customer_df["FN"].map(map2)
customer_df["Active"] = customer_df["Active"].map(map2)

map3 =  {"LEFT CLUB":0, "PRE-CREATE": 1, "ACTIVE":2} #np.nan:0
customer_df["club_member_status"] = customer_df["club_member_status"].map(map3)

#drop nan values
customer_df = customer_df.dropna(subset=["age", "fashion_news_frequency", "club_member_status"], axis=0)

# check for any nan values
percent_c = (customer_df.isnull().sum()/customer_df.isnull().count()*100).sort_values(ascending = False)

# Encode customer id and postal code columns
num_customers = customer_df['customer_id'].nunique()

Customer_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_customers+1).fit(customer_df[['customer_id']].to_numpy().reshape(-1, 1))
customer_df['customer_id'] = Customer_id_Encoder.transform(customer_df[['customer_id']].to_numpy().reshape(-1, 1))


num_postal = customer_df['postal_code'].nunique()

postal_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_postal+1).fit(customer_df[['postal_code']].to_numpy().reshape(-1, 1))
customer_df['postal_code'] = postal_code_Encoder.transform(customer_df[['postal_code']].to_numpy().reshape(-1, 1))




# we cannot use FN and Active. They have a lot of NaN values. We keep the rest
df_c = customer_df[['customer_id','age', 'club_member_status', 'postal_code', 'fashion_news_frequency']]



# Preprocess article dataframe
articles_df = pd.read_csv(data_path+'articles.csv')

articles_df = articles_df.drop(columns=["product_code", "product_type_no", 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no','index_code','index_group_no','section_no','garment_group_no'], axis=1)

# Encode
num_products = articles_df['article_id'].nunique()

article_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['article_id']].to_numpy().reshape(-1, 1))
articles_df['article_id'] = article_id_Encoder.transform(articles_df[['article_id']].to_numpy().reshape(-1, 1))


articles_df = articles_df.drop(columns=["index_name", "section_name", "product_group_name", "garment_group_name", "perceived_colour_value_name", "perceived_colour_master_name"], axis=1)



num_departments = articles_df['department_name'].nunique()
num_colours = articles_df['colour_group_name'].nunique()
num_prod_names = articles_df['prod_name'].nunique()
num_prod_type_names = articles_df['product_type_name'].nunique()
num_graphical = articles_df['graphical_appearance_name'].nunique()
num_index = articles_df['index_group_name'].nunique()




Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(articles_df[['department_name']].to_numpy().reshape(-1, 1))
Prod_name_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_names+1).fit(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
Prod_type_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_type_names+1).fit(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
Graphical_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_graphical+1).fit(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
Index_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_index+1).fit(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))





articles_df['colour_group_name'] = Colour_Encoder.transform(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
articles_df['department_name'] = Department_encoder.transform(articles_df[['department_name']].to_numpy().reshape(-1, 1))
articles_df['prod_name'] = Prod_name_encoder.transform(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
articles_df['product_type_name'] = Prod_type_encoder.transform(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
articles_df['graphical_appearance_name'] = Graphical_encoder.transform(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
articles_df['index_group_name'] = Index_encoder.transform(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))




articles_df[['article_id','prod_name','product_type_name','graphical_appearance_name','colour_group_name','department_name','index_group_name']].to_csv('Data/Preprocessed/article_df_numeric.csv',index=False)

# From the articles we see several columns with the same information. section_name, product_group_name and garment_group_name gives almost the same information just with differnet headlines.
# we can discard the garment_group_name and product_group_name and the number associated. 
# we can also discard all the colour codes and only keep the colour_group_name.
# The detailed desciption is also discarded.
# Difference between product_code and product_type_no is product_code gives the number for the same item, while product_type_no is the number for all the trousers fx. 



# check for NaN
percent_a = (articles_df.isnull().sum()/articles_df.isnull().count()*100).sort_values(ascending = False)





# Preprocess transaction train dataframe
transactions_df = pd.read_csv(data_path+'transactions_train.csv')

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






