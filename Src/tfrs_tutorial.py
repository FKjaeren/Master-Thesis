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
u_colour = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["colour_group_name"]))))




class UserModel(tf.keras.Model):

  def __init__(self):
    super().__init__()


    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_customer, mask_token=None),
        tf.keras.layers.Embedding(len(u_customer) + 1, 32),
    ])
    self.age_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_age, mask_token=None),
        tf.keras.layers.Embedding(len(u_age) + 1, 32),
    ])
    self.colour_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_colour, mask_token=None),
        tf.keras.layers.Embedding(len(u_colour) + 1, 32),
    ])
    self.prod_group_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_prod_group, mask_token=None),
        tf.keras.layers.Embedding(len(u_prod_group) + 1, 32),
    ])


  def call(self, inputs):
    #return tf.concat([
    return self.user_embedding(inputs["customer_id"])
        #self.age_embedding(inputs["age"]),
        #self.colour_embedding(inputs["colour_group_name"]),
        #self.prod_group_embedding(inputs["product_group_name"]),
        #tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
    #], axis=1)

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, layer_sizes):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

class MovieModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    #max_tokens = 10_000

    self.title_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
          vocabulary=u_articles, mask_token=None),
      tf.keras.layers.Embedding(len(u_articles) + 1, 32)
    ])
    """ 
    self.title_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens)

    self.title_text_embedding = tf.keras.Sequential([
      self.title_vectorizer,
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

    self.title_vectorizer.adapt(articles) """

  def call(self, titles):
    #return tf.concat([
    return self.title_embedding(titles)
       # self.title_text_embedding(titles),
    #], axis=1)

class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, layer_sizes):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.embedding_model = MovieModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)


class MovielensModel(tfrs.models.Model):

  def __init__(self, layer_sizes):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    self.candidate_model = CandidateModel(layer_sizes)
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(64).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "customer_id": features["customer_id"]
    })
    movie_embeddings = self.candidate_model(features["prod_name"])

    return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)

class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      UserModel(),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      MovieModel(),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "customer_id": features["customer_id"],
        "age": features["age"],
        "colour_group_name": features["colour_group_name"],
        "product_group_name": features["product_group_name"],

    })
    movie_embeddings = self.candidate_model(features["prod_name"])

    return self.task(query_embeddings, movie_embeddings)

tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()


model = MovielensModel([64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

one_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=10,
    verbose=0)

accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")


num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [(x + 1)* 5 for x in range(num_validation_runs)]

plt.plot(epochs, one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="1 layer")
#plt.plot(epochs, two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="2 layers")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy");
plt.legend()
plt.show()



model = MovielensModel([128, 64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

three_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=3,
    verbose=0)

accuracy = three_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")





model.fit(cached_train, epochs=10)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")
