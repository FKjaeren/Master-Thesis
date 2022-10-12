import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
import copy


customers = pd.read_csv('Data/Raw/customers.csv')


map =  {"None":0, "NONE":0, "Regularly": 1, "Monthly":2, np.nan:0}
customers["fashion_news_frequency"] = customers["fashion_news_frequency"].map(map)

map2 =  {np.nan:0, 1.0: 1}
customers["FN"] = customers["FN"].map(map2)
customers["Active"] = customers["Active"].map(map2)

map3 =  {"LEFT CLUB":0, "PRE-CREATE": 1, "ACTIVE":2, np.nan:0}
customers["club_member_status"] = customers["club_member_status"].map(map3)

customers["age"] = customers["age"].fillna(0)




num_customers = customers['customer_id'].nunique()

Customer_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_customers+1).fit(customers[['customer_id']].to_numpy().reshape(-1, 1))

customers['customer_id'] = Customer_id_Encoder.transform(customers[['customer_id']].to_numpy().reshape(-1, 1))

num_postal = customers['postal_code'].nunique()

postal_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_postal+1).fit(customers[['postal_code']].to_numpy().reshape(-1, 1))
customers['postal_code'] = postal_code_Encoder.transform(customers[['postal_code']].to_numpy().reshape(-1, 1))



#################################################################
###
# Articles
articles_df = pd.read_csv('Data/Raw/articles.csv')


#articles_df = articles_df.drop(columns=["product_code", "product_type_no", 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no','index_code','index_group_no','section_no','garment_group_no'], axis=1)
articles_df = articles_df.drop(columns=["detail_desc"], axis=1)

# Encode all string columns

num_products = articles_df['article_id'].nunique()

num_product_code = articles_df['product_code'].nunique()
num_product_type_no = articles_df['product_type_no'].nunique()
num_graphical_appearance_no = articles_df['graphical_appearance_no'].nunique()
num_colour_group_code = articles_df['colour_group_code'].nunique()
num_perceived_colour_value_id = articles_df['perceived_colour_value_id'].nunique()
num_perceived_colour_master_id = articles_df['perceived_colour_master_id'].nunique()
num_department_no = articles_df['department_no'].nunique()
num_index_code = articles_df['index_code'].nunique()
num_index_group_no = articles_df['index_group_no'].nunique()
num_section_no = articles_df['section_no'].nunique()
num_garment_group_no = articles_df['garment_group_no'].nunique()
num_index_name = articles_df['index_name'].nunique()
num_section_name = articles_df['section_name'].nunique()
num_product_group_name = articles_df['product_group_name'].nunique()
num_garment_group_name = articles_df['garment_group_name'].nunique()
num_perceived_colour_value_name = articles_df['perceived_colour_value_name'].nunique()
num_perceived_colour_master_name = articles_df['perceived_colour_master_name'].nunique()

num_departments = articles_df['department_name'].nunique()
num_colours = articles_df['colour_group_name'].nunique()
num_prod_names = articles_df['prod_name'].nunique()
num_prod_type_names = articles_df['product_type_name'].nunique()
num_graphical = articles_df['graphical_appearance_name'].nunique()
num_index = articles_df['index_group_name'].nunique()

article_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['article_id']].to_numpy().reshape(-1, 1))

product_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['product_code']].to_numpy().reshape(-1, 1))
product_type_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['product_type_no']].to_numpy().reshape(-1, 1))
graphical_appearance_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['graphical_appearance_no']].to_numpy().reshape(-1, 1))
colour_group_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['colour_group_code']].to_numpy().reshape(-1, 1))
perceived_colour_value_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['perceived_colour_value_id']].to_numpy().reshape(-1, 1))
perceived_colour_master_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['perceived_colour_master_id']].to_numpy().reshape(-1, 1))
department_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['department_no']].to_numpy().reshape(-1, 1))
index_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['index_code']].to_numpy().reshape(-1, 1))
index_group_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['index_group_no']].to_numpy().reshape(-1, 1))
section_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['section_no']].to_numpy().reshape(-1, 1))
garment_group_no_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['garment_group_no']].to_numpy().reshape(-1, 1))
index_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['index_name']].to_numpy().reshape(-1, 1))
section_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['section_name']].to_numpy().reshape(-1, 1))
product_group_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['product_group_name']].to_numpy().reshape(-1, 1))
garment_group_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['garment_group_name']].to_numpy().reshape(-1, 1))
perceived_colour_value_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['perceived_colour_value_name']].to_numpy().reshape(-1, 1))
perceived_colour_master_name_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['perceived_colour_master_name']].to_numpy().reshape(-1, 1))

Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(articles_df[['department_name']].to_numpy().reshape(-1, 1))
Prod_name_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_names+1).fit(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
Prod_type_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_type_names+1).fit(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
Graphical_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_graphical+1).fit(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
Index_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_index+1).fit(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))

articles_df['article_id'] = article_id_Encoder.transform(articles_df[['article_id']].to_numpy().reshape(-1, 1))
articles_df['product_code'] = product_code_Encoder.transform(articles_df[['product_code']].to_numpy().reshape(-1, 1))
articles_df['product_type_no'] = product_type_no_Encoder.transform(articles_df[['product_type_no']].to_numpy().reshape(-1, 1))
articles_df['graphical_appearance_no'] = graphical_appearance_no_Encoder.transform(articles_df[['graphical_appearance_no']].to_numpy().reshape(-1, 1))
articles_df['colour_group_code'] = colour_group_code_Encoder.transform(articles_df[['colour_group_code']].to_numpy().reshape(-1, 1))
articles_df['perceived_colour_value_id'] = perceived_colour_value_id_Encoder.transform(articles_df[['perceived_colour_value_id']].to_numpy().reshape(-1, 1))
articles_df['perceived_colour_master_id'] = perceived_colour_master_id_Encoder.transform(articles_df[['perceived_colour_master_id']].to_numpy().reshape(-1, 1))
articles_df['department_no'] = department_no_Encoder.transform(articles_df[['department_no']].to_numpy().reshape(-1, 1))
articles_df['index_code'] = index_code_Encoder.transform(articles_df[['index_code']].to_numpy().reshape(-1, 1))
articles_df['index_group_no'] = index_group_no_Encoder.transform(articles_df[['index_group_no']].to_numpy().reshape(-1, 1))
articles_df['section_no'] = section_no_Encoder.transform(articles_df[['section_no']].to_numpy().reshape(-1, 1))
articles_df['garment_group_no'] = garment_group_no_Encoder.transform(articles_df[['garment_group_no']].to_numpy().reshape(-1, 1))
articles_df['index_name'] = index_name_Encoder.transform(articles_df[['index_name']].to_numpy().reshape(-1, 1))
articles_df['section_name'] = section_name_Encoder.transform(articles_df[['section_name']].to_numpy().reshape(-1, 1))
articles_df['product_group_name'] = product_group_name_Encoder.transform(articles_df[['product_group_name']].to_numpy().reshape(-1, 1))
articles_df['garment_group_name'] = garment_group_name_Encoder.transform(articles_df[['garment_group_name']].to_numpy().reshape(-1, 1))
articles_df['perceived_colour_value_name'] = perceived_colour_value_name_Encoder.transform(articles_df[['perceived_colour_value_name']].to_numpy().reshape(-1, 1))
articles_df['perceived_colour_master_name'] = perceived_colour_master_name_Encoder.transform(articles_df[['perceived_colour_master_name']].to_numpy().reshape(-1, 1))
articles_df['colour_group_name'] = Colour_Encoder.transform(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
articles_df['department_name'] = Department_encoder.transform(articles_df[['department_name']].to_numpy().reshape(-1, 1))
articles_df['prod_name'] = Prod_name_encoder.transform(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
articles_df['product_type_name'] = Prod_type_encoder.transform(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
articles_df['graphical_appearance_name'] = Graphical_encoder.transform(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
articles_df['index_group_name'] = Index_encoder.transform(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))




percent_a = (articles_df.isnull().sum()/articles_df.isnull().count()*100).sort_values(ascending = False)



######################################################


# Transactions data


transactions_df_original = pd.read_csv('Data/Raw/transactions_train.csv')
transactions_df = copy.deepcopy(transactions_df_original)

transactions_df = transactions_df[transactions_df['t_dat']>'2019-09-21']

#transactions_df = transactions_df.iloc[-300000:].reset_index().drop(['index'],axis = 1)

#datetime and create a day, month and year column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)

transactions_df['day'] =  pd.DatetimeIndex(transactions_df['t_dat']).day
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df['year'] =  pd.DatetimeIndex(transactions_df['t_dat']).year
transactions_df = transactions_df.drop(['t_dat'], axis = 1)

# Divide into train, valid and test
splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))
train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]


num_days = 31
num_months = 12
num_year = train.year.nunique()

Year_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_year+1).fit(train[['year']])

transactions_df['year'] = Year_encoder.transform(transactions_df[['year']].to_numpy().reshape(-1, 1))



# Create season column
transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 


# Encode customer, price and article and map values in season
transactions_df['customer_id'] = Customer_id_Encoder.transform(transactions_df[['customer_id']])
transactions_df['article_id'] = article_id_Encoder.transform(transactions_df[['article_id']])

train['price'] = train['price'].round(decimals=4)
transactions_df['price'] = transactions_df['price'].round(decimals=4)
num_prices = train['price'].nunique()
Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(train[['price']])
transactions_df['price'] = Price_encoder.transform(transactions_df[['price']])

map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}

transactions_df['season'] = transactions_df['season'].map(map_season)


train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]




train = train.merge(articles_df, on = 'article_id', how = 'left')
train = train.merge(customers, how = 'left', on ='customer_id')



from scipy.stats import chi2_contingency




def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))



rows= []

for var1 in train:
  col = []
  for var2 in train:
    print(train[[var1]].columns)
    print(train[[var2]].columns)
    cramers =cramers_V(train[var1], train[var2]) # Cramer's V test
    col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
  rows.append(col)
  
cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns = train.columns, index =train.columns)



df




valid = valid.merge(articles_df, on = 'article_id', how = 'left')
valid = valid.merge(customers, how = 'left', on ='customer_id')

test = test.merge(articles_df, on = 'article_id', how = 'left')
test = test.merge(customers, how = 'left', on ='customer_id')
