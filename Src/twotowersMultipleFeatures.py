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




interactions_dict = X[["customer_id", "age", "prod_name"]]

interactions_dict = {name: np.array(value) for name, value in interactions_dict.items()}
interactions = tf.data.Dataset.from_tensor_slices(interactions_dict)

articles_dict = X[['prod_name']].drop_duplicates()
articles_dict = {name: np.array(value) for name, value in articles_dict.items()}
articles = tf.data.Dataset.from_tensor_slices(articles_dict)

## map the features in interactions and items
interactions = interactions.map(lambda x: {
                              'customer_id' : (x['customer_id']), 
                              'age' : int(x['age']),
                              'prod_name' : (x['prod_name'])})

articles = articles.map(lambda x: str(x['prod_name']))




u_articles = np.unique(np.concatenate(list(articles.batch(1000))))
u_customer = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["customer_id"]))))
u_age = np.unique(np.concatenate(list(interactions.batch(1_000).map(lambda x: x["age"]))))


##########################
# Trying to create dict tensorflow dataset
#X['article_id'] = X['article_id'].astype(str)


X[['customer_id',      
          'prod_name',
          'age',
         ]] = X[['customer_id','prod_name', 'age']].astype(str)
ds = tf.data.Dataset.from_tensor_slices(dict(X))
""" 
articles_dict = X[['prod_name']].drop_duplicates()
articles_dict = {name: np.array(value) for name, value in articles_dict.items()}
articles = tf.data.Dataset.from_tensor_slices(articles_dict)

articles = articles.batch(1_000).map(lambda x: x['prod_name'])

u_articles = np.unique(np.concatenate(list(articles))) """

ds = ds.map(lambda x: {
    #"article_id": (x["article_id"]),
    "prod_name": (x["prod_name"]),
    "customer_id": (x["customer_id"]),
    "age": (x["age"]),
    #"product_group_name": x["product_group_name"],
})
""" title_text = tf.keras.layers.TextVectorization()
title_text.adapt(ds.map(lambda x: x["prod_name"]))
for row in ds.batch(1_000).map(lambda x: x["prod_name"]).take(1):
  print(title_text(row))
title_text.get_vocabulary()[0:10] """

#p_name = ds.batch(1_000).map(lambda x: x["product_group_name"])
#articles = ds.batch(1_000).map(lambda x: x["prod_name"])
articles = ds.batch(1_000).map(lambda x: x['prod_name'])
customer = ds.batch(1_000).map(lambda x: x["customer_id"])
age = ds.batch(1_000).map(lambda x: x["age"])


u_articles = np.unique(np.concatenate(list(articles)))
u_customer = np.unique(np.concatenate(list(customer)))
u_age = np.unique(np.concatenate(list(age)))
u_age

for x in articles.take(1).as_numpy_iterator():
  pprint.pprint(x)

    


class CustomerModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.customer_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_customer, mask_token=None),
        tf.keras.layers.Embedding(len(u_customer) + 1, 32),
    ])
    self.age_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_age, mask_token=None),
        tf.keras.layers.Embedding(len(u_age) + 1, 32),
    ])

  def call(self, inputs):
      # Take the input dictionary, pass it through each input layer,
      # and concatenate the result.
      return tf.concat([
          self.customer_embedding(inputs["customer_id"]),
          self.age_embedding(inputs["age"]),], axis=1)


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes, projection_dim=None):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = CustomerModel()
            
#         self.dense_layers = tf.keras.Sequential([
#                                     tfrs.layers.dcn.Cross(projection_dim=projection_dim,
#                                                           kernel_initializer="glorot_uniform"),
#                                     tf.keras.layers.Dense(256, activation="relu"),
#                                     tf.keras.layers.Dense(128, activation="relu"),
#                                     tf.keras.layers.Dense(1)
#             ])

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential(tfrs.layers.dcn.Cross(projection_dim=projection_dim,
                                        kernel_initializer="glorot_uniform"))

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class ArticleModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        
        self.embedding_dimension = 32

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
          tf.keras.layers.experimental.preprocessing.StringLookup(
              vocabulary=u_articles,mask_token=None),
          tf.keras.layers.Embedding(len(u_articles) + 1, self.embedding_dimension)
        ])

        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
          self.title_vectorizer,
          tf.keras.layers.Embedding(max_tokens, self.embedding_dimension, mask_zero=True),
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(articles)

    def call(self, inputs):
        print('features object looks like: ', inputs[:,0])

        return tf.concat([
            self.title_embedding(inputs),
            self.title_text_embedding(inputs), #[:,0]
        ], axis=1)


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes, projection_dim=None):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = ArticleModel()

         # Then construct the layers.
        self.dense_layers = tf.keras.Sequential(tfrs.layers.dcn.Cross(projection_dim=projection_dim,
                                                kernel_initializer="glorot_uniform"))

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)



class CrossDNNModel(tfrs.models.Model):

    def __init__(self, layer_sizes, rating_weight: float, retrieval_weight: float, projection_dim=None ):
        super().__init__()
        
        self.query_model : tf.keras.Model = QueryModel(layer_sizes)
        self.candidate_model : tf.keras.Model = CandidateModel(layer_sizes)
        
        ## rating and retrieval task.
        
        self.rating_task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
                 
        self.retrieval_task : tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=articles.batch(128).map(self.candidate_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

    def compute_loss(self, features, training=False):
        
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        #ratings = features.pop("quantity")
        
        query_embeddings = self.query_model({
            "customer_id": features["customer_id"],
            "age": features["age"],
        })
    
        article_embeddings = self.candidate_model(features['prod_name'])       
        retrieval_loss = self.retrieval_task(query_embeddings, article_embeddings)
    
    
        return self.retrieval_task(query_embeddings, article_embeddings)


class RetrievalModel(tfrs.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32

    # Set up a model for representing articles.
    self.article_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_articles, mask_token=None),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(u_articles) + 1, embedding_dimension)
    ])

    self.customer_model = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=u_customer, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(u_customer) + 1, embedding_dimension)
        ])

    # Set up a task to optimize the model and compute metrics.
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=articles.batch(128).cache().map(self.article_model)
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    #print('features object looks like: ', features[:,0])
    # We pick out the customer features and pass them into the customer model.
    customer_embeddings = self.customer_model(features["customer_id"])
    print('customer_embedding looks like: ', customer_embeddings)
    # And pick out the article features and pass them into the article model,
    # getting embeddings back.
    positive_article_embeddings = self.article_model(features["article_id"])
    print('postitive_article_embedding looks like: ', positive_article_embeddings)

    # The task computes the loss and the metrics.

    return self.task(customer_embeddings, positive_article_embeddings, compute_metrics=not training)



tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

model = CrossDNNModel([32], rating_weight=0.5, retrieval_weight=0.5, projection_dim=None)

model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, validation_data=cached_test,
        validation_freq=5, epochs=3)

train_accuracy = model.evaluate(
    train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")





