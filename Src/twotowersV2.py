from turtle import pos
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv('Data/Preprocessed/transactions_df_subset.csv')


# Preprocess customer dataframe
# check for NaN and make a subset with relevant columns
percent_c = (customer_df.isnull().sum()/customer_df.isnull().count()*100).sort_values(ascending = False)
df_c = customer_df[['customer_id','age', 'club_member_status']]
df_c.dropna(subset=['age'])


# Preprocess article dataframe
# subset relevant columns
df_a = articles_df[['article_id','product_type_no', 'prod_name', 'product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'section_name']]
# check for NaN
percent_a = (df_a.isnull().sum()/df_a.isnull().count()*100).sort_values(ascending = False)

# Preprocess transaction train dataframe
#datetime and create a month column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df['year'] =  pd.DatetimeIndex(transactions_df['t_dat']).year
transactions_df['day'] =  pd.DatetimeIndex(transactions_df['t_dat']).day



transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 


X = transactions_df.merge(df_a, how = "left", on = "article_id")

X = pd.get_dummies(X, columns = ['season','product_type_name','product_group_name','section_name'])

X = X.merge(df_c[['customer_id','age']], how = 'left', on = 'customer_id')

splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

train = X.iloc[:splitrange]
valid = X.iloc[splitrange+1:splitrange2]
test = X.iloc[splitrange2:]

#df_a_sub = df_a.drop([])
articles_sub = articles_df[['article_id']].values.flatten()
customers_sub = customer_df[['customer_id']].values.flatten()


#articles_sub = articles[['article_id']].values.flatten()
#customers_sub = customers[['customer_id']].values.flatten()


u_customer = customer_df.customer_id.unique()
u_article = articles_df.article_id.astype(str).unique()

#CUDA_VISIBLE_DEVICES=""
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# we create a tensor shape we can pass to the model for all articles
article_ds = tf.data.Dataset.from_tensor_slices(dict(articles_df[['article_id']]))
articles = article_ds.map(lambda x: x['article_id'])
# Reetrieval for article and customer ids
class RetrievalModel(tfrs.Model):
  
  def __init__(self, num_features):
    self.num_features = num_features
    super().__init__()
    embedding_dimension = 32

    # Set up a model for representing articles.
    self.article_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_article, mask_token=None, dtype=tf.int32),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(u_article)+1, embedding_dimension, input_length = num_features)
    ])

    self.customer_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_customer, mask_token=None, dtype=tf.int32),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(u_customer)+1,embedding_dimension, input_length= num_features)
        ])

    # Set up a task to optimize the model and compute metrics.
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=articles.batch(128).cache().map(self.article_model)
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    customer_id, data = tuple(zip(*features))
    customer_id = tf.data.Dataset.from_tensor_slices(customer_id)
    data = tf.data.Dataset.from_tensor_slices(data)
    #self.num_features = num_features
    #print('features object looks like: ', features[:,0])
    # We pick out the customer features and pass them into the customer model.
    customer_embeddings = self.customer_model(customer_id)
    #print('customer_embedding looks like: ', customer_embeddings)
    # And pick out the article features and pass them into the article model,
    # getting embeddings back.
    positive_article_embeddings = self.article_model(data)
    #print('postitive_article_embedding looks like: ', positive_article_embeddings)

    # The task computes the loss and the metrics.

    return self.task(customer_embeddings, positive_article_embeddings)#, compute_metrics=not training


# wE need to create tensor 
#train_ds = tf.data.Dataset.from_tensor_slices(dict(train[['customer_id','article_id']]))
#print(train_ds.shape())
#train_ds_v2=tf.convert_to_tensor(train[['customer_id','article_id']])
#dataVar_tensor = tf.constant(train[['customer_id','article_id']], shape=train[['customer_id','article_id']].shape)
#dataset = tf.data.Dataset.from_tensor_slices((dataVar_tensor))
#dataset = dataset.batch(1000)
#test_ds = tf.data.Dataset.from_tensor_slices(dict(test[['customer_id','article_id']]))
num_epochs = 3

features = train.drop(['customer_id','t_dat','prod_name','department_name','colour_group_name'],axis=1).columns
feature_data = train.drop(['customer_id','t_dat','prod_name','department_name','colour_group_name'],axis=1)
num_features = int(len(features)+1)

model = RetrievalModel(num_features)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1), loss='None')

"""
class Mapper():
    def __init__(self, possible_articles, num_negative_articles):
        self.num_possible_articles = len(possible_articles)
        self.possible_articles_tensor = tf.constant(possible_articles, dtype=tf.int32)
        self.num_negative_articles = num_negative_articles
        self.y = tf.one_hot(0, num_negative_articles+1)
    def __call__(self, customer, article):
        random_negatives_indexes  = tf.random.uniform((self.num_negative_articles,),minval = 0, maxval = self.num_possible_articles , dtype = tf.int32) 
        negative_products =  tf.gather(self.possible_articles_tensor, random_negatives_indexes)
        candidates = tf.concat([article, negative_products], axis = 0)
        return (customer, candidates), self.y
"""
def get_dataset(df):
    dummy_customer_tensor = tf.constant(df[['customer_id']].values, dtype =tf.string)
    article_tensor = tf.constant(feature_data.values,dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    #dataset = dataset.map(Mapper(articles, number_negative_articles)) 
    #dataset = dataset.batch(1024)
    return dataset, dummy_customer_tensor, article_tensor

dataset, customer_tensor, article_tensor = get_dataset(train)

for customer_id, data in dataset.take(1):  # only take first element of dataset
    customer_id = customer_id.numpy()
    data = data.numpy()

model.fit(dataset, epochs=num_epochs)

test_dataset = get_dataset(test)

model.evaluate(test_dataset, return_dict=True)


scann = tfrs.layers.factorized_top_k.ScaNN(model.customer_model, k=5)
scann.index_from_dataset(
    tf.data.Dataset.zip((articles, articles.map(model.article_model)))
)