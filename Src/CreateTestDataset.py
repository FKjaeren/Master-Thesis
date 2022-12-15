import torch
import pandas as pd
import pickle
from CreateNegativeSamples import CreateNegativeSamples


test_df = pd.read_csv('Data/Preprocessed/test_df_subset.csv')
train_df = pd.read_csv('Data/Preprocessed/train_df_subset.csv')

articles_df = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv')
customer_df = pd.read_csv('Data/Preprocessed/customer_df_numeric_subset.csv')


test_df_subset = test_df.sample(15000, random_state=42)

test_df_subset.to_csv('../../../../../../work3/s174478/Data/test_dataset_subset.csv', index = False)
del test_df

data_split1 = round(0.15*len(test_df_subset['customer_id']))
data_split2 = round(0.30*len(test_df_subset['customer_id']))
data_split3 = round(0.45*len(test_df_subset['customer_id']))
data_split4 = round(0.60*len(test_df_subset['customer_id']))
data_split5 = round(0.75*len(test_df_subset['customer_id']))
data_split6 = round(0.90*len(test_df_subset['customer_id']))

test_split1 = test_df_subset.iloc[:data_split1]
test_split2 = test_df_subset.iloc[data_split1+1:data_split2]
test_split3 = test_df_subset.iloc[data_split2+1:data_split3]
test_split4 = test_df_subset.iloc[data_split3+1:data_split4]
test_split5 = test_df_subset.iloc[data_split4+1:data_split5]
test_split6 = test_df_subset.iloc[data_split5+1:data_split6]
test_split7 = test_df_subset.iloc[data_split6+1:]

del test_df_subset
num_products = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv').nunique()[0]
print("Test0")


negative_df_test = CreateNegativeSamples(test_split1, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")
test_with_negative = test_split1.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split1)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')
test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

print("Test1")
test_with_negative_final = test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name', 'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']]
del (test_with_negative)

print("directory and variables after first dataframe is created: ",dir())

## 1. part of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split2, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")
test_with_negative = test_split2.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split2)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')
test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

test_with_negative_final = pd.concat([test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']], test_with_negative_final])
del (test_with_negative)

print("directory and variables after second dataframe is created: ",dir())

## 2. part of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split3, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")
test_with_negative = test_split3.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split3)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')
test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

test_with_negative_final = pd.concat([test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']], test_with_negative_final])
del (test_with_negative)

print("directory and variables after thrid dataframe is created: ",dir())

## 3. part of the dataset of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split4, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")

test_with_negative = test_split4.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split4)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')

test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

test_with_negative_final = pd.concat([test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']], test_with_negative_final])
del (test_with_negative)

print("directory and variables after fourth dataframe is created: ",dir())

test_with_negative_final.to_csv('../../../../../../work3/s174478/Data/test_dataset_with_negative_part1.csv', index = False)

del (test_with_negative_final)


## 4. part of the dataset of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split5, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")

test_with_negative = test_split5.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split5)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')

test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

test_with_negative_final = test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name', 'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']]
del (test_with_negative)

## 5. part of the dataset of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split6, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")

test_with_negative = test_split6.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)
del (test_split6)
print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')

test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

test_with_negative_final = pd.concat([test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']], test_with_negative_final])
del (test_with_negative)

##6. part of the dataset of the dataframe is done

negative_df_test = CreateNegativeSamples(test_split7, train_df, num_products, type_df='Test', method = 'Random_choices')
print("Negative samples have been made.")
del (train_df)

test_with_negative = test_split7.merge(negative_df_test, how = 'outer', on = ['customer_id','article_id','price','sales_channel_id','day','month','year','season']).fillna(0).drop('negative_values',axis=1)

del (test_split7)

print("The two large dataframes have been merged :)")
del (negative_df_test)
test_with_negative = test_with_negative[['customer_id','article_id','price','sales_channel_id','day','month','year','season','target']]
test_with_negative = test_with_negative.merge(articles_df, on = 'article_id', how = 'left')

del (articles_df)

test_with_negative = test_with_negative.merge(customer_df, how = 'left', on ='customer_id')

del (customer_df)

test_with_negative_final = pd.concat([test_with_negative[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name',
    'graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status',
    'fashion_news_frequency', 'age', 'postal_code','target']], test_with_negative_final])
del (test_with_negative)

## 7. part of the dataset of the dataframe is done


test_with_negative_final.to_csv('../../../../../../work3/s174478/Data/test_dataset_with_negative_part2.csv', index = False)
del (test_with_negative_final)
