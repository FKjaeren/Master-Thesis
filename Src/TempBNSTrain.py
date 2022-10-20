import numpy as np
import pandas as pd
import pickle
import torch

train_df = pd.read_csv('Data/Preprocessed/train_df.csv')[0:600000]
valid_df = pd.read_csv('Data/Preprocessed/valid_df.csv')
test_df = pd.read_csv('Data/Preprocessed/test_df.csv')

with open(r"Data/Preprocessed/number_uniques_dict.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

u_count = number_uniques_dict['n_customers']
i_count = number_uniques_dict['n_products']

u_list = pd.read_csv('Data/Preprocessed/customer_df_numeric.csv')['customer_id']
i_list = pd.read_csv('Data/Preprocessed/article_df_numeric.csv')[['article_id']]

popularity = np.zeros(i_count-1)

popularity_temp = train_df[['article_id']].value_counts().sort_index().to_frame().reset_index().rename({'article_id':'article_id',0:'counts'},axis = 1)
popularity_temp = i_list.merge(popularity_temp, how = 'left', on = 'article_id').fillna(0)

for i in range(i_count):
    popularity[i] = popularity_temp.loc[i].values[0]

popularity[:] = popularity_temp['counts']


dict_negative_items = {}
for u in u_list:
    positive_items = set(train_df[train_df['customer_id']==u]['article_id'])
    negative_items = set(i_list['article_id'])-positive_items
    dict_negative_items[u] = list(negative_items)