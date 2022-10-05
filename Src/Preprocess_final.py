import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
import copy
import pickle

data_path = 'Data/Raw/'

customer_df = pd.read_csv(data_path+'customers.csv')
articles_df = pd.read_csv(data_path+'articles.csv')
transactions_df_original = pd.read_csv(data_path+'transactions_train.csv')

articles_df = articles_df.drop(columns=["product_code", "product_type_no", 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 
                                        'perceived_colour_master_id', 'department_no','index_code','index_group_no','section_no','garment_group_no'], axis=1)

articles_df = articles_df.drop(columns=["index_name", "section_name", "product_group_name", "garment_group_name", "perceived_colour_value_name", "perceived_colour_master_name"], axis=1)

transactions_df_enriched = transactions_df_original.merge(customer_df, how = 'left', on = 'customer_id')

transactions_df_enriched = transactions_df_enriched.merge(articles_df, how = 'left', on = 'article_id')

num_customers = customer_df['customer_id'].nunique()
num_postal = customer_df['postal_code'].nunique()
num_departments = articles_df['department_name'].nunique()
num_colours = articles_df['colour_group_name'].nunique()
num_prod_names = articles_df['prod_name'].nunique()
num_prod_type_names = articles_df['product_type_name'].nunique()
num_graphical = articles_df['graphical_appearance_name'].nunique()
num_index = articles_df['index_group_name'].nunique()
num_products = articles_df['article_id'].nunique()

postal_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_postal+1).fit(customer_df[['postal_code']].to_numpy().reshape(-1, 1))
Customer_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_customers+1).fit(customer_df[['customer_id']].to_numpy().reshape(-1, 1))


article_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['article_id']].to_numpy().reshape(-1, 1))
Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(articles_df[['department_name']].to_numpy().reshape(-1, 1))
Prod_name_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_names+1).fit(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
Prod_type_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_type_names+1).fit(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
Graphical_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_graphical+1).fit(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
Index_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_index+1).fit(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))

map =  {"None":0, "NONE":0, "Regularly": 1, "Monthly":2, np.nan:0}
transactions_df_enriched["fashion_news_frequency"] = transactions_df_enriched["fashion_news_frequency"].map(map)

map2 =  {np.nan:0, 1.0: 1}
transactions_df_enriched["FN"] = transactions_df_enriched["FN"].map(map2)
transactions_df_enriched["Active"] = transactions_df_enriched["Active"].map(map2)

map3 =  {"LEFT CLUB":0, "PRE-CREATE": 1, "ACTIVE":2, np.nan:0}
transactions_df_enriched["club_member_status"] = transactions_df_enriched["club_member_status"].map(map3)

transactions_df_enriched["age"] = transactions_df_enriched["age"].fillna(0)


#datetime and create a day, month and year column
transactions_df_enriched.t_dat = pd.to_datetime(transactions_df_enriched.t_dat)

transactions_df_enriched['day'] =  pd.DatetimeIndex(transactions_df_enriched['t_dat']).day
transactions_df_enriched['month'] =  pd.DatetimeIndex(transactions_df_enriched['t_dat']).month
transactions_df_enriched['year'] =  pd.DatetimeIndex(transactions_df_enriched['t_dat']).year
transactions_df_enriched = transactions_df_enriched.drop(['t_dat'], axis = 1)

# Divide into train, valid and test
splitrange = round(0.75*len(transactions_df_enriched['customer_id']))
splitrange2 = round(0.95*len(transactions_df_enriched['customer_id']))
train = transactions_df_enriched.iloc[:splitrange]

num_days = transactions_df_enriched.day.nunique()
num_months = transactions_df_enriched.month.nunique()
num_year = transactions_df_enriched.year.nunique()

Year_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_year+1).fit(train[['year']])
train['price'] = train['price'].round(decimals=4)
transactions_df_enriched['price'] = transactions_df_enriched['price'].round(decimals=4)
num_prices = train['price'].nunique()
Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(train[['price']])
map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}

transactions_df_enriched.loc[(transactions_df_enriched['month']>= 1) & (transactions_df_enriched['month'] <=2), 'season'] = 'Winter'
transactions_df_enriched.loc[(transactions_df_enriched['month'] == 12), 'season'] = 'Winter' 
transactions_df_enriched.loc[(transactions_df_enriched['month'] >= 3) & (transactions_df_enriched['month'] <=5), 'season'] = 'Spring' 
transactions_df_enriched.loc[(transactions_df_enriched['month'] >= 6) & (transactions_df_enriched['month'] <=8),'season'] = 'Summer' 
transactions_df_enriched.loc[(transactions_df_enriched['month'] >= 9) & (transactions_df_enriched['month'] <=11), 'season'] = 'Autumn' 


num_sales_channels = transactions_df_enriched.sales_channel_id.nunique()


number_uniques_dict = {'n_customers' : num_customers+1, 'n_products':num_products+1, 'n_departments':num_departments+1, 'n_colours': num_colours+1, 'n_prod_names' : num_prod_names+1,
                        'n_prod_type_names': num_prod_type_names, 'n_graphical':num_graphical, 'n_index' : num_index, 'n_postal':num_postal, 'n_fashion_news_frequency': 3+1, 'n_FN' : 2+1, 
                        'n_active':2+1, 'n_club_member_status':3+1 ,'n_prices':num_prices, 'n_seasons': 4+1, 'n_sales_channels' : num_sales_channels+1, 'n_days' : num_days+1,
                        'n_months' : num_months +1, 'n_year': num_year+1}


with open(r"Data/Preprocessed/number_uniques_dict.pickle", "wb") as output_file:
    pickle.dump(number_uniques_dict, output_file)

transactions_df_enriched['season'] = transactions_df_enriched['season'].map(map_season)

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
pickle.dump(Year_encoder, open('Models/Year_Encoder.sav', 'wb'))


# Encode customer, price and article and map values in season
transactions_df_enriched['customer_id'] = Customer_id_Encoder.transform(transactions_df_enriched[['customer_id']].to_numpy().reshape(-1, 1))
transactions_df_enriched['article_id'] = article_id_Encoder.transform(transactions_df_enriched[['article_id']].to_numpy().reshape(-1, 1))
transactions_df_enriched['colour_group_name'] = Colour_Encoder.transform(transactions_df_enriched[['colour_group_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['department_name'] = Department_encoder.transform(transactions_df_enriched[['department_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['prod_name'] = Prod_name_encoder.transform(transactions_df_enriched[['prod_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['product_type_name'] = Prod_type_encoder.transform(transactions_df_enriched[['product_type_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['graphical_appearance_name'] = Graphical_encoder.transform(transactions_df_enriched[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['index_group_name'] = Index_encoder.transform(transactions_df_enriched[['index_group_name']].to_numpy().reshape(-1, 1))
transactions_df_enriched['year'] = Year_encoder.transform(transactions_df_enriched[['year']].to_numpy().reshape(-1, 1))
transactions_df_enriched['price'] = Price_encoder.transform(transactions_df_enriched[['price']].to_numpy().reshape(-1, 1))
transactions_df_enriched['postal_code'] = postal_code_Encoder.transform(transactions_df_enriched[['postal_code']].to_numpy().reshape(-1, 1))

transactions_df_enriched = transactions_df_enriched.drop(['detail_desc'], axis = 1)

transactions_df_enriched.to_csv('Data/Preprocessed/AllDataOneTable.csv', index=False)