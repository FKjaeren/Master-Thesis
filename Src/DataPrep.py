from copy import deepcopy
import numpy as np 
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy
from Src.CreateNegativeSamples import CreateNegativeSamples
#import seaborn as sns

#from Src.PreprocessIdData import Article_id_encoder, Price_encoder

data_path = 'Data/Raw/'

customer_df = pd.read_csv(data_path+'customers_subset.csv')
customer_df.head()
customer_df = customer_df.drop(["FN", "Active"], axis = 1)

########### Preprocess customer dataframe
# check for NaN and map values in some of the columns 
#customer_df = customer_df[customer_df.fashion_news_frequency != "None"]

map =  {"None":0, "NONE":0, "Regularly": 1, "Monthly":2, np.nan:0}
customer_df["fashion_news_frequency"] = customer_df["fashion_news_frequency"].map(map)

#map2 =  {np.nan:0, 1.0: 1}
#customer_df["FN"] = customer_df["FN"].map(map2)
#customer_df["Active"] = customer_df["Active"].map(map2)

map3 =  {"LEFT CLUB":0, "PRE-CREATE": 1, "ACTIVE":2, np.nan:0}
customer_df["club_member_status"] = customer_df["club_member_status"].map(map3)

customer_df["age"] = customer_df["age"].fillna(0)

#drop nan values
#customer_df = customer_df.dropna(subset=["age", "fashion_news_frequency", "club_member_status"], axis=0)

# check for any nan values
percent_c = (customer_df.isnull().sum()/customer_df.isnull().count()*100).sort_values(ascending = False)

# Encode customer id and postal code columns

num_customers = customer_df['customer_id'].nunique()

Customer_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_customers+1).fit(customer_df[['customer_id']].to_numpy().reshape(-1, 1))

customer_df['customer_id'] = Customer_id_Encoder.transform(customer_df[['customer_id']].to_numpy().reshape(-1, 1))

num_postal = customer_df['postal_code'].nunique()

postal_code_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_postal+1).fit(customer_df[['postal_code']].to_numpy().reshape(-1, 1))
customer_df['postal_code'] = postal_code_Encoder.transform(customer_df[['postal_code']].to_numpy().reshape(-1, 1))

customer_df.to_csv('Data/Preprocessed/customer_df_numeric_subset.csv',index=False)



# we cannot use FN and Active. They have a lot of NaN values. We keep the rest
#df_c = customer_df[['customer_id','age', 'club_member_status', 'postal_code', 'fashion_news_frequency']]



################### Preprocess article dataframe
articles_df = pd.read_csv(data_path+'articles_subset.csv')
# Drop columns with information we already have from columns alike and drop columns with numbers for groups or colours etc.

# From the articles we see several columns with the same information. section_name, 
# product_group_name and garment_group_name gives almost the same information just with differnet headlines.
# we can discard the garment_group_name and product_group_name and the number associated. 
# we can also discard all the colour codes and only keep the colour_group_name.
# The detailed desciption is also discarded.
# Difference between product_code and product_type_no is product_code gives the number for the same item, while product_type_no is the number for all the trousers fx.
articles_df = articles_df.drop(columns=["product_code", "product_type_no", 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no','index_code','index_group_no','section_no','garment_group_no'], axis=1)

# Encode all string columns
num_products = articles_df['article_id'].nunique()

article_id_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_products+1).fit(articles_df[['article_id']].to_numpy().reshape(-1, 1))
articles_df['article_id'] = article_id_Encoder.transform(articles_df[['article_id']].to_numpy().reshape(-1, 1))

#articles_df = articles_df.drop(columns=["index_name", "section_name", "product_group_name", "garment_group_name", "perceived_colour_value_name", "perceived_colour_master_name"], axis=1)


articles_df = articles_df.drop(columns=["index_group_name", "product_type_name","index_name", "section_name", "product_group_name", "garment_group_name", "perceived_colour_value_name", "perceived_colour_master_name"], axis=1)

num_departments = articles_df['department_name'].nunique()
num_colours = articles_df['colour_group_name'].nunique()
num_prod_names = articles_df['prod_name'].nunique()
#num_prod_type_names = articles_df['product_type_name'].nunique()
num_graphical = articles_df['graphical_appearance_name'].nunique()
#num_index = articles_df['index_group_name'].nunique()


Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(articles_df[['department_name']].to_numpy().reshape(-1, 1))
Prod_name_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_names+1).fit(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
#Prod_type_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prod_type_names+1).fit(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
Graphical_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_graphical+1).fit(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
#Index_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_index+1).fit(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))



articles_df['colour_group_name'] = Colour_Encoder.transform(articles_df[['colour_group_name']].to_numpy().reshape(-1, 1))
articles_df['department_name'] = Department_encoder.transform(articles_df[['department_name']].to_numpy().reshape(-1, 1))
articles_df['prod_name'] = Prod_name_encoder.transform(articles_df[['prod_name']].to_numpy().reshape(-1, 1))
#articles_df['product_type_name'] = Prod_type_encoder.transform(articles_df[['product_type_name']].to_numpy().reshape(-1, 1))
articles_df['graphical_appearance_name'] = Graphical_encoder.transform(articles_df[['graphical_appearance_name']].to_numpy().reshape(-1, 1))
#articles_df['index_group_name'] = Index_encoder.transform(articles_df[['index_group_name']].to_numpy().reshape(-1, 1))


# Save the csv file wihtout the detalied descritption if we need later
#articles_df[['article_id','prod_name','product_type_name','graphical_appearance_name','colour_group_name','department_name','index_group_name']].to_csv('Data/Preprocessed/article_df_numeric_subset.csv',index=False)
articles_df[['article_id','prod_name','graphical_appearance_name','colour_group_name', 'department_name']].to_csv('Data/Preprocessed/article_df_numeric_subset.csv',index=False)

############################# Preprocess transaction train dataframe
transactions_df_original = pd.read_csv(data_path+'transactions_train_subset.csv')
transactions_df = copy.deepcopy(transactions_df_original)

transactions_df = transactions_df[transactions_df['t_dat']>'2019-09-21']

#transactions_df = transactions_df.iloc[-300000:].reset_index().drop(['index'],axis = 1)

ones_data = np.ones(shape=(len(transactions_df),1))
target = pd.DataFrame(ones_data, columns=['targets'])

#transactions_df = pd.concat([transactions_df,target], axis = 1)
transactions_df['target'] = target['targets']
#datetime and create a day, month and year column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)

transactions_df['day'] =  pd.DatetimeIndex(transactions_df['t_dat']).day
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df['year'] =  pd.DatetimeIndex(transactions_df['t_dat']).year
transactions_df = transactions_df.drop(['t_dat'], axis = 1)
transactions_df['price'] = transactions_df['price'].round(decimals=4)

# Divide into train, valid and test
splitrange = round(0.8*len(transactions_df['customer_id']))
splitrange2 = round(0.975*len(transactions_df['customer_id']))
train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]


num_days = 31
num_months = 12
num_year = transactions_df.year.nunique()

Year_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_year+1).fit(train[['year']].to_numpy().reshape(-1, 1))
train['year'] = Year_encoder.transform(train[['year']].to_numpy().reshape(-1, 1))
valid['year'] = Year_encoder.transform(valid[['year']].to_numpy().reshape(-1, 1))
test['year'] = Year_encoder.transform(test[['year']].to_numpy().reshape(-1, 1))
transactions_df['year'] = Year_encoder.transform(transactions_df[['year']])



# Create season column
transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 


# Encode customer, price and article and map values in season
transactions_df['customer_id'] = Customer_id_Encoder.transform(transactions_df[['customer_id']].to_numpy().reshape(-1, 1))
transactions_df['article_id'] = article_id_Encoder.transform(transactions_df[['article_id']].to_numpy().reshape(-1, 1))

num_prices = train['price'].nunique()
Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(train[['price']].to_numpy().reshape(-1, 1))
transactions_df['price'] = Price_encoder.transform(transactions_df[['price']].to_numpy().reshape(-1, 1))

#train['customer_id'] = Customer_id_Encoder.transform(train[['customer_id']].to_numpy().reshape(-1, 1))
#train['article_id'] = article_id_Encoder.transform(train[['article_id']].to_numpy().reshape(-1, 1))
map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}

transactions_df['season'] = transactions_df['season'].map(map_season)

num_sales_channels = transactions_df.sales_channel_id.nunique()

#number_uniques_dict = {'n_customers' : num_customers+1, 'n_products':num_products+1, 'n_departments':num_departments+1, 'n_colours': num_colours+1, 'n_prod_names' : num_prod_names+1,
#                        'n_prod_type_names': num_prod_type_names, 'n_graphical':num_graphical, 'n_index' : num_index, 'n_postal':num_postal, 'n_fashion_news_frequency': 3+1, 'n_FN' : 2+1, 
#                        'n_active':2+1, 'n_club_member_status':3+1 ,'n_prices':num_prices, 'n_seasons': 4+1, 'n_sales_channels' : num_sales_channels+1, 'n_days' : num_days+1,
#                        'n_months' : num_months +1, 'n_year': num_year+1}


number_uniques_dict = {'n_customers' : num_customers+1, 'n_products':num_products+1, 'n_departments':num_departments+1, 'n_colours': num_colours+1, 'n_prod_names' : num_prod_names+1,
                         'n_graphical':num_graphical, 'n_postal':num_postal, 'n_fashion_news_frequency': 3+1, 'n_FN' : 2+1, 
                        'n_active':2+1, 'n_club_member_status':3+1 ,'n_prices':num_prices, 'n_seasons': 4+1, 'n_sales_channels' : num_sales_channels+1, 'n_days' : num_days+1,
                        'n_months' : num_months +1, 'n_year': num_year+1}

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "wb") as output_file:
    pickle.dump(number_uniques_dict, output_file)


# Pickle dump the used encoder for later use
pickle.dump(Customer_id_Encoder, open('Models/Customer_Id_Encoder_subset.sav', 'wb'))
pickle.dump(article_id_Encoder, open('Models/Article_Id_Encoder_subset.sav', 'wb'))
pickle.dump(Price_encoder, open('Models/Price_Encoder_subset.sav', 'wb'))
pickle.dump(Colour_Encoder, open('Models/Colour_Encoder_subset.sav', 'wb'))
pickle.dump(Department_encoder, open('Models/Department_Encoder_subset.sav', 'wb'))
pickle.dump(Prod_name_encoder, open('Models/Prod_Name_Encoder_subset.sav', 'wb'))
#pickle.dump(Index_encoder, open('Models/Index_Encoder_subset.sav', 'wb'))
pickle.dump(Graphical_encoder, open('Models/Graphical_Encoder_subset.sav', 'wb'))
#pickle.dump(Prod_type_encoder, open('Models/Prod_Type_Encoder_subset.sav', 'wb'))
pickle.dump(postal_code_Encoder, open('Models/Postal_Code_Encoder_subset.sav', 'wb'))
pickle.dump(Year_encoder, open('Models/Year_Encoder_subset.sav', 'wb'))


def GetPreprocessedDF(transactions_df = transactions_df, n_negative_samples = 10,Method = 'FM'):
    if(Method == 'FM'):
        splitrange = round(0.8*len(transactions_df['customer_id']))
        splitrange2 = round(0.975*len(transactions_df['customer_id']))

        train = transactions_df.iloc[:splitrange]
        valid = transactions_df.iloc[splitrange+1:splitrange2]
        test = transactions_df.iloc[splitrange2:]

        negative_df = CreateNegativeSamples(train, train, n_negative_samples, type_df = 'Train', method = 'Random_choices')

        train = train.merge(negative_df, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
        train = train[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]


        train = train.merge(articles_df, on = 'article_id', how = 'left').drop(['detail_desc'], axis = 1)
        train = train.merge(customer_df, how = 'left', on ='customer_id')

        negative_df_valid = CreateNegativeSamples(valid, train, n_negative_samples, type_df = 'Train', method = 'Random_choices')
        
        valid = valid.merge(negative_df_valid, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
        valid = valid[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]

        valid = valid.merge(articles_df, on = 'article_id', how = 'left').drop(['detail_desc'], axis = 1)
        valid = valid.merge(customer_df, how = 'left', on ='customer_id')


        #negative_df_test = CreateNegativeSamples(test, train, num_products, type_df='Test', method = 'Random_choices')

        #test_with_negative = test.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
        #test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
        #test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left').drop(['detail_desc'], axis = 1)
        #test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

        test = test[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
        test = test.merge(articles_df, on = 'article_id', how = 'left').drop(['detail_desc'], axis = 1)
        test = test.merge(customer_df, how = 'left', on ='customer_id')

        """ train = train[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 'product_type_name',
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'index_group_name', 'FN', 'Active', 'club_member_status',
            'fashion_news_frequency', 'age', 'postal_code','target']]

        valid = valid[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 'product_type_name',
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'index_group_name', 'FN', 'Active', 'club_member_status',
            'fashion_news_frequency', 'age', 'postal_code','target']]

        test = test[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 'product_type_name',
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'index_group_name', 'FN', 'Active', 'club_member_status',
            'fashion_news_frequency', 'age', 'postal_code','target']] """

        train = train[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 
            'graphical_appearance_name', 'colour_group_name', 'department_name', 'club_member_status',
            'fashion_news_frequency', 'age', 'postal_code','target']]

        valid = valid[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'club_member_status', 'fashion_news_frequency', 'age', 'postal_code','target']]

        test = test[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'club_member_status', 'fashion_news_frequency', 'age', 'postal_code','target']]

        #test_with_negative = test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            #'month', 'year', 'season', 'prod_name', 'product_type_name',
            #'graphical_appearance_name', 'colour_group_name', 'department_name',
            #'index_group_name', 'FN', 'Active', 'club_member_status',
            #'fashion_news_frequency', 'age', 'postal_code','target']]

        train.to_csv('Data/Preprocessed/train_df_subset.csv', index = False)
        valid.to_csv('Data/Preprocessed/valid_df_subset.csv', index=False)
        test.to_csv('Data/Preprocessed/test_df_subset.csv', index = False)
        #test_with_negative.to_csv('Data/Preprocessed/test_with_negative_subset.csv', index = False)
        print('Dataframes for a Factorization Machine model have been saved')


    elif(Method == 'MF'):
        ### customer_id, article_id, price, sales_channel_id, season, day, month, year
        ### prod_name, product_type_name, graphical_appearance_name, 'colour_group_name, department_name, index_group_name
        ### FN, Active, club_member_status, fashion_news_frequency, age, postal_code

        # Find the mean price of each article in transactions df and decode price
        article_id_aggregated = transactions_df[['article_id','price']].groupby(by = 'article_id').mean().reset_index()
        article_id_aggregated['price'] = Price_encoder.transform(article_id_aggregated[['price']])
        # Find the most used features for each article in transactions df
        Most_frequent_sales_channel = transactions_df.groupby('article_id')['sales_channel_id'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_season = transactions_df.groupby('article_id')['season'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_day = transactions_df.groupby('article_id')['day'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_month = transactions_df.groupby('article_id')['month'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_year = transactions_df.groupby('article_id')['year'].apply(lambda x: x.value_counts().index[0]).reset_index()

        # Merge with the customer df
        transactions_df_enriched = transactions_df.merge(customer_df, how = 'left', on = 'customer_id')
        # Get he mean age for each article
        article_id_aggregated_v2 = transactions_df_enriched[['article_id','age']].groupby('article_id').mean().reset_index()
        # Find the most used features from customer for each article in customer df
        #Most_frequent_FN = transactions_df_enriched.groupby('article_id')['FN'].apply(lambda x: x.value_counts().index[0]).reset_index()
        #Most_frequent_Active = transactions_df_enriched.groupby('article_id')['Active'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_club_member_status = transactions_df_enriched.groupby('article_id')['club_member_status'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_fashion_news_frequency = transactions_df_enriched.groupby('article_id')['fashion_news_frequency'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_postal_code = transactions_df_enriched.groupby('article_id')['postal_code'].apply(lambda x: x.value_counts().index[0]).reset_index()

        # Merge all dataframe so we have one dataframe with preprossed numerical features for articles and all most frequent features
        Product_preprocessed_model_df = transactions_df[['article_id']]
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(articles_df.drop(['detail_desc'], axis = 1), how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(article_id_aggregated, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_sales_channel, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_season, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_day, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_month, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_year, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(article_id_aggregated_v2, how='left', on = 'article_id')
        #Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_FN, how = 'left', on = 'article_id')
        #Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_Active, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_club_member_status, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_fashion_news_frequency, how = 'left', on = 'article_id')
        Product_preprocessed_model_df = Product_preprocessed_model_df.merge(Most_frequent_postal_code, how = 'left', on = 'article_id')

        Product_df_preprocessed = deepcopy(articles_df)
        Product_df_preprocessed = Product_df_preprocessed.merge(article_id_aggregated, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_sales_channel, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_season, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_day, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_month, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_year, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(article_id_aggregated_v2, how='left', on = 'article_id')
        #Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_FN, how = 'left', on = 'article_id')
        #Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_Active, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_club_member_status, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_fashion_news_frequency, how = 'left', on = 'article_id')
        Product_df_preprocessed = Product_df_preprocessed.merge(Most_frequent_postal_code, how = 'left', on = 'article_id')

        # Save the merged dataframe
        Product_preprocessed_model_df.to_csv('Data/Preprocessed/FinalProductDataFrame_subset.csv', index = False)
        Product_df_preprocessed.to_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts_subset.csv', index = False)

        # We do the same for all customer
        customer_id_aggregated = transactions_df_enriched[['customer_id','price']].groupby('customer_id').mean().reset_index()
        customer_id_aggregated['price'] = Price_encoder.transform(customer_id_aggregated[['price']])
        Most_frequent_sales_channel = transactions_df.groupby('customer_id')['sales_channel_id'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_season = transactions_df.groupby('customer_id')['season'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_day = transactions_df.groupby('customer_id')['day'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_month = transactions_df.groupby('customer_id')['month'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_year = transactions_df.groupby('customer_id')['year'].apply(lambda x: x.value_counts().index[0]).reset_index()

        transactions_df_enriched = transactions_df.merge(articles_df, how = 'left', on = 'article_id')

        # Most used features for all customers
        Most_frequent_prod_name = transactions_df_enriched.groupby('customer_id')['prod_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
        #Most_frequent_product_type_name = transactions_df_enriched.groupby('customer_id')['product_type_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_graphical_appearance_name = transactions_df_enriched.groupby('customer_id')['graphical_appearance_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_colour_group_name = transactions_df_enriched.groupby('customer_id')['colour_group_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
        Most_frequent_department_name = transactions_df_enriched.groupby('customer_id')['department_name'].apply(lambda x: x.value_counts().index[0]).reset_index()
        #Most_frequent_index_group_name = transactions_df_enriched.groupby('customer_id')['index_group_name'].apply(lambda x: x.value_counts().index[0]).reset_index()

        # Merge all dataframe so we have one dataframe with preprossed numerical features for customer and all most frequent features
        Customer_preprocessed_model_df = transactions_df[['customer_id']]
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(customer_df, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(customer_id_aggregated, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_sales_channel, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_season, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_day, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_month, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_year, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_prod_name, how = 'left', on = 'customer_id')
        #Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_product_type_name, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_graphical_appearance_name, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_colour_group_name, how = 'left', on = 'customer_id')
        Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_department_name, how = 'left', on = 'customer_id')
        #Customer_preprocessed_model_df = Customer_preprocessed_model_df.merge(Most_frequent_index_group_name, how = 'left', on = 'customer_id')
        # Save it
        Customer_preprocessed_model_df.to_csv('Data/Preprocessed/FinalCustomerDataFrame_subset.csv', index = False)

        # Save the transactions df aswell
        #transactions_df['price'] = Price_encoder.transform(transactions_df[['price']])
        #transactions_df.to_csv('Data/Preprocessed/transactions_df_numeric.csv', index = False)
        print('Dataframes for a Matrix Factorization model have been saved')


## Call the "GetPreprocessedDF" function with parameter: "method == 'FM'" to get dataframes for a Factorization machine model.
## Call the "GetPreprocessedDF" function with parameter: "method == 'MF'" to get dataframes for a Matrix Factorization model.
num_negative_samples = 10
GetPreprocessedDF(transactions_df,n_negative_samples=num_negative_samples, Method = 'FM')