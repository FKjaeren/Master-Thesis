import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Dict, Text
import pprint

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
import warnings
warnings.filterwarnings("ignore")
data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')
transactions_df = transactions_df[transactions_df['t_dat'] >='2020-08-01']


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

##################################
# Defining vocabularys for categorical features aka article and customer
article_ds = tf.data.Dataset.from_tensor_slices(dict(df_a[['article_id']]))
articles = article_ds.map(lambda x: x['article_id'])

u_article = df_a.article_id.astype(str).unique()

Article_lookup = tf.keras.layers.StringLookup(vocabulary=u_article, mask_token=None, dtype=tf.int32)
#Article_lookup.adapt(articles.map(lambda x: x["article_id"]))
print(f"Vocabulary: {Article_lookup.get_vocabulary()[:3]}")
Article_lookup(["108775015", "108775044"])

num_hashing_bins = 200_000

article_hashing = tf.keras.layers.Hashing(
    num_bins=num_hashing_bins
)

article_hashing(["108775015", "108775044"])

#####
# Customer

u_customer = df_c.customer_id.unique()

Customer_lookup = tf.keras.layers.StringLookup(vocabulary = u_customer)

customer_embedding = tf.keras.layers.Embedding(Customer_lookup.vocab_size(), 32)






################3 
# Now text features
prod_name_ds = tf.data.Dataset.from_tensor_slices(dict(df_a[['prod_name']]))
prod = prod_name_ds.map(lambda x: x['prod_name'])


u_prod = df_a.prod_name.unique()

title_text = tf.keras.layers.TextVectorization(vocabulary=u_prod)


for row in df_a.batch(1).map(lambda x: x["prod_name"]).take(1):
  print(title_text(row))


title_text.get_vocabulary()[40:45]


# we create a tensor shape we can pass to the model for all articles
article_ds2 = tf.data.Dataset.from_tensor_slices(dict(df_a[['article_id']]))
articles2 = article_ds2.map(lambda x: x['article_id'])




features = train.drop(['customer_id','t_dat','prod_name','department_name','colour_group_name'],axis=1).columns
feature_data = train.drop(['customer_id','t_dat','prod_name','department_name','colour_group_name'],axis=1)
num_features = int(len(features)+1)

def get_dataset(df):
    dummy_customer_tensor = tf.constant(df[['customer_id']].values, dtype =tf.string)
    article_tensor = tf.constant(feature_data.values,dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    #dataset = dataset.map(Mapper(articles, number_negative_articles)) 
    #dataset = dataset.batch(1024)
    return dataset 
dataset = get_dataset(train)

for x in dataset.take(1).as_numpy_iterator():
  pprint.pprint(x)

class ArticleModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.article_embedding = tf.keras.Sequential([
      Article_lookup,
      tf.keras.layers.Embedding(Article_lookup.vocab_size(), 32)
    ])
    self.title_text_embedding = tf.keras.Sequential([
      tf.keras.layers.TextVectorization(max_tokens=max_tokens),
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      # We average the embedding of individual words to get one embedding vector
      # per title.
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

  def call(self, inputs):
    return tf.concat([
        self.article_embedding(inputs["article_id"]),
        self.title_text_embedding(inputs["prod_name"]),
    ], axis=1)


class CustomerModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.customer_embedding = tf.keras.Sequential([
        Customer_lookup,
        tf.keras.layers.Embedding(Customer_lookup.vocab_size(), 32),
    ])



class RetrievalModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      CustomerModel(),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      ArticleModel(),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    print('features object looks like: ', features[:,0])

    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "customer_id": features["customer_id"]
    })
    movie_embeddings = self.candidate_model(features["article_id"])

    return self.task(query_embeddings, movie_embeddings)

model = RetrievalModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))



dataVar_tensor = tf.constant(train[['customer_id','article_id']], shape=train[['customer_id','article_id']].shape)
dataset = tf.data.Dataset.from_tensor_slices((dataVar_tensor))
dataset = dataset.batch(1000)

num_epochs = 3

model.fit(dataset, epochs=num_epochs)


model = MovielensModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

one_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")

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
    return dataset 

dataset = get_dataset(train)

for customer_id, data in dataset.take(1):  # only take first element of dataset
    customer_id = customer_id.numpy()
    data = data.numpy()

model.fit(dataset, epochs=num_epochs)


model.evaluate(test_ds, return_dict=True)