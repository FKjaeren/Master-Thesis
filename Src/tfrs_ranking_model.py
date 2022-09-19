from code import interact
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Dict, Text
import pprint


import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

plt.style.use('seaborn-whitegrid')
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')
transactions_df = transactions_df[transactions_df['t_dat'] >='2020-09-01']



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


X = transactions_df.merge(df_a, how = "left", on = "article_id")


X = X.merge(df_c[['customer_id','age']], how = 'left', on = 'customer_id')


""" 
interactions_dict = X[["customer_id", "age", "prod_name", "colour_group_name", "product_group_name"]].astype(str)




interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

articles_dict = X[['prod_name']].drop_duplicates()
articles_dict = {name: np.array(value) for name, value in articles_dict.items()}
articles = tf.data.Dataset.from_tensor_slices(articles_dict)



## map the features in interactions and items
interactions = interactions.map(lambda x: {
                              'customer_id' : (x['customer_id']), 
                              'age' : (x['age']),
                              'prod_name' : (x['prod_name']),
                              'product_group_name' : (x['product_group_name']),
                              'colour_group_name' : (x['colour_group_name'])
                              })

articles = articles.map(lambda x: (x['prod_name']))


for x in articles.take(100).as_numpy_iterator():
  pprint.pprint(x)
u_articles = np.unique(np.concatenate(list(articles.batch(1_000))))
u_customer = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["customer_id"]))))
u_age = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["age"]))))
u_prod_group = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["product_group_name"]))))
u_colour = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["colour_group_name"])))) """


da = np.ones(len(X))



X["quantity"] = da 

X[["customer_id", "prod_name"]] = X[["customer_id", "prod_name"]].astype(str)
X[["quantity"]] = X[["quantity"]].astype(float)



interactions_dict = X.groupby(['customer_id', 'prod_name'])['quantity'].sum().reset_index()




interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

articles_dict = X[['prod_name']].drop_duplicates()
articles_dict = {name: np.array(value) for name, value in articles_dict.items()}
articles = tf.data.Dataset.from_tensor_slices(articles_dict)



## map the features in interactions and items
interactions = interactions.map(lambda x: {
                              'customer_id' : (x['customer_id']), 
                              'prod_name' : (x['prod_name']),    
                              'quantity' : (x['quantity']),
                              })

articles = articles.map(lambda x: (x['prod_name']))


for x in articles.take(100).as_numpy_iterator():
  pprint.pprint(x)
u_articles = np.unique(np.concatenate(list(articles.batch(1_000))))
u_customer = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["customer_id"]))))



class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=u_customer, mask_token=None),
          tf.keras.layers.Embedding(len(u_customer) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=u_articles, mask_token=None),
          tf.keras.layers.Embedding(len(u_articles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
          # Learn multiple dense layers.
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          # Make rating predictions in the final layer.
          tf.keras.layers.Dense(1)
  ])

    def call(self, inputs):

        customer_id, prod_name = inputs

        user_embedding = self.user_embeddings(customer_id)
        movie_embedding = self.movie_embeddings(prod_name)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


class RetailModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("quantity")

        rating_predictions = self.ranking_model(
            (features["customer_id"], features["prod_name"]))

        # The task computes the loss and the metrics.


        return self.task(labels = labels, predictions=rating_predictions)


model = RetailModel()

model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.5))


tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(8192).cache()


cached_test = test.batch(4096).cache()



model.fit(cached_train, epochs=100)

model.evaluate(cached_test, return_dict=True)






















class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_customer, mask_token=None),
      tf.keras.layers.Embedding(len(u_customer) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_articles, mask_token=None),
      tf.keras.layers.Embedding(len(u_articles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    customer_id, prod_name = inputs

    user_embedding = self.user_embeddings(customer_id)
    movie_embedding = self.movie_embeddings(prod_name)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))


RankingModel()((["0001d44dbe7f6c4b35200abdb052c77a87596fe1bdcc37e011580a479e80aa94"], ["Victoria dress"]))


class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["customer_id"], features["prod_name"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("quantity")

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)

model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


model.fit(cached_train, epochs=30)

model.evaluate(cached_test, return_dict=True)


test_ratings = {}
test_movie_titles = ["Matey", "Clara jogger", "Anna"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({
      "customer_id": np.array(["0005ed68483efa39644c45185550a82dd09acb07622acb17cff1304ed649f077"]),
      "prod_name": np.array([movie_title])
  })

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")