





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





class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32

    # User and movie models.
    self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_articles, mask_token=None),
      tf.keras.layers.Embedding(len(u_articles) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_customer, mask_token=None),
      tf.keras.layers.Embedding(len(u_customer) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=articles.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["customer_id"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features["prod_name"])

    return (
        user_embeddings,
        movie_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("quantity")

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)



tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(8192).cache()


cached_test = test.batch(4096).cache()


model = MovielensModel(rating_weight=0.7, retrieval_weight=0.3)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))



model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")