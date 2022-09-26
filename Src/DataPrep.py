
from copy import deepcopy
import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy

import seaborn as sns

#from Src.PreprocessIdData import Article_id_encoder, Price_encoder

data_path = 'Data/Raw/'

customer_df = pd.read_csv(data_path+'customers.csv')
num_customers = customer_df['customer_id'].nunique()

Customer_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_customers+1).fit(customer_df[['customer_id']].to_numpy().reshape(-1, 1))


# Preprocess customer dataframe
# check for NaN and map values in some of the columns 
#customer_df
#customer_df = customer_df[customer_df.fashion_news_frequency != "None"]

map =  {"None":0, "NONE":0, "Regularly": 1, "Monthly":2, np.nan:0}
customer_df["fashion_news_frequency"] = customer_df["fashion_news_frequency"].map(map)

map2 =  {np.nan:0, 1.0: 1}
customer_df["FN"] = customer_df["FN"].map(map2)
customer_df["Active"] = customer_df["Active"].map(map2)

map3 =  {"LEFT CLUB":0, "PRE-CREATE": 1, "ACTIVE":2, np.nan:0}
customer_df["club_member_status"] = customer_df["club_member_status"].map(map3)

#drop nan values
#customer_df = customer_df.dropna(subset=["age", "fashion_news_frequency", "club_member_status"], axis=0)

# check for any nan values
percent_c = (customer_df.isnull().sum()/customer_df.isnull().count()*100).sort_values(ascending = False)

# Encode customer id and postal code columns

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
transactions_df_original = pd.read_csv(data_path+'transactions_train.csv')
transactions_df = copy.deepcopy(transactions_df_original)


splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))
train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]

#datetime and create a month column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)

transactions_df['day'] =  pd.DatetimeIndex(transactions_df['t_dat']).day
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df['year'] =  pd.DatetimeIndex(transactions_df['t_dat']).year
transactions_df = transactions_df.drop(['t_dat'], axis = 1)

transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 
# drop month column

transactions_df['customer_id'] = Customer_id_Encoder.transform(transactions_df[['customer_id']])
transactions_df['article_id'] = article_id_Encoder.transform(transactions_df[['article_id']])

train['price'] = train['price'].round(decimals=4)
transactions_df['price'] = transactions_df['price'].round(decimals=4)
num_prices = train['price'].nunique()
Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(train[['price']])

map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}

transactions_df['season'] = transactions_df['season'].map(map_season)


pickle.dump(Customer_id_Encoder, open('Models/Customer_Id_Encoder.sav', 'wb'))
pickle.dump(article_id_Encoder, open('Models/Article_Id_Encoder.sav', 'wb'))
pickle.dump(Price_encoder, open('Models/Price_Encoder.sav', 'wb'))
pickle.dump(Colour_Encoder, open('Models/Colour_Encoder.sav', 'wb'))
pickle.dump(Department_encoder, open('Models/Department_Encoder.sav', 'wb'))
pickle.dump(Prod_name_encoder, open('Models/Prod_Name_Encoder.sav', 'wb'))
pickle.dump(Index_encoder, open('Models/Index_Encoder.sav', 'wb'))
pickle.dump(Graphical_encoder, open('Models/Graphical_Encoder.sav', 'wb'))
pickle.dump(Prod_type_encoder, open('Models/Prod_Type_Encoder.sav', 'wb'))
pickle.dump(postal_code_Encoder, open('Models/Postal_Code_Encoder.sav', 'wb'))


### customer_id, article_id, price, sales_channel_id, season, day, month, year
### prod_name, product_type_name, graphical_appearance_name, 'colour_group_name, department_name, index_group_name
### FN, Active, club_member_status, fashion_news_frequency, age, postal_code

article_id_aggregated = transactions_df[['article_id','price']].groupby(by = 'article_id').mean().reset_index()
article_id_aggregated['price'] = Price_encoder.transform(article_id_aggregated[['price']])
Most_frequent_sales_channel = transactions_df.groupby('article_id')['sales_channel_id'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_season = transactions_df.groupby('article_id')['season'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_day = transactions_df.groupby('article_id')['day'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_month = transactions_df.groupby('article_id')['month'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_year = transactions_df.groupby('article_id')['year'].apply(lambda x: x.value_counts().index[0]).reset_index()

transactions_df_enriched = transactions_df.merge(customer_df, how = 'left', on = 'customer_id')

article_id_aggregated_v2 = transactions_df_enriched[['article_id','age']].groupby('article_id').mean().reset_index()

Most_frequent_FN = transactions_df_enriched.groupby('article_id')['FN'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_Active = transactions_df_enriched.groupby('article_id')['Active'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_club_member_status = transactions_df_enriched.groupby('article_id')['club_member_status'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_fashion_news_frequency = transactions_df_enriched.groupby('article_id')['fashion_news_frequency'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_postal_code = transactions_df_enriched.groupby('article_id')['postal_code'].apply(lambda x: x.value_counts().index[0]).reset_index()

Product_preprocessed_model_df = transactions_df[['article_id']]
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(articles_df.drop(['detail_desc'], axis = 1), how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(article_id_aggregated, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_sales_channel, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_season, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_day, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_month, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_year, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(article_id_aggregated_v2, how='left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_FN, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_Active, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_club_member_status, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_fashion_news_frequency, how = 'left', on = 'article_id')
Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_postal_code, how = 'left', on = 'article_id')

Product_preprocessed_model_df.to_csv('Data/Preprocessed/FinalProductDataFrame.csv', index = False)

customer_id_aggregated = transactions_df_enriched[['customer_id','price']].groupby('customer_id').mean().reset_index()
customer_id_aggregated['price'] = Price_encoder.transform(customer_id_aggregated[['price']])
Most_frequent_sales_channel = transactions_df.groupby('customer_id')['sales_channel_id'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_season = transactions_df.groupby('customer_id')['season'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_day = transactions_df.groupby('customer_id')['day'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_month = transactions_df.groupby('customer_id')['month'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_year = transactions_df.groupby('customer_id')['year'].apply(lambda x: x.value_counts().index[0]).reset_index()

transactions_df_enriched = transactions_df.merge(articles_df, how = 'left', on = 'customer_id')

Most_frequent_prod_name = transactions_df_enriched.groupby('article_id')['prod_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_product_type_name = transactions_df_enriched.groupby('article_id')['product_type_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_graphical_appearance_name = transactions_df_enriched.groupby('article_id')['graphical_appearance_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_colour_group_name = transactions_df_enriched.groupby('article_id')['colour_group_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_department_name = transactions_df_enriched.groupby('article_id')['department_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
Most_frequent_index_group_name = transactions_df_enriched.groupby('article_id')['index_group_name'].apply(lambda x: x.value_counts().index[0]).reset_index()


Customer_preprocessed_model_df = transactions_df[['customer_id']]
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(customer_df, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(customer_id_aggregated, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_sales_channel, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_season, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_day, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_month, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_year, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_prod_name, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_product_type_name, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_graphical_appearance_name, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_colour_group_name, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_department_name, how = 'left', on = 'customer_id')
Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_index_group_name, how = 'left', on = 'customer_id')

Customer_preprocessed_model_df.to_csv('Data/Preprocessed/FinalCustomerDataFrame.csv', index = False)


transactions_df['price'] = Price_encoder.transform(transactions_df[['price']])
transactions_df.to_csv('Data/Preprocessed/transactions_df_numeric.csv', index = False)
