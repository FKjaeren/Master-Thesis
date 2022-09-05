from turtle import pos
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

transactions_df = pd.read_csv('Data/Raw/transactions_train.csv')
transactions_df = transactions_df[transactions_df['t_dat'] >='2020-08-01']




articles = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
# somehow it doesnot work when it its integers, so we set it to str
articles['article_id'] = articles['article_id'].astype(str)
transactions_df['article_id'] = transactions_df['article_id'].astype(str)

#articles_sub = articles[['article_id']].values.flatten()
#customers_sub = customers[['customer_id']].values.flatten()


u_customer = customers.customer_id.unique()
u_article = articles.article_id.unique()

# we create a tensor shape we can pass to the model for all articles
article_ds = tf.data.Dataset.from_tensor_slices(dict(articles[['article_id']]))
articles = article_ds.map(lambda x: x['article_id'])
# Reetrieval for article and customer ids
class RetrievalModel(tfrs.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32

    # Set up a model for representing articles.
    self.article_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=u_article, mask_token=None),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(u_article) + 1, embedding_dimension)
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
    print('features object looks like: ', features[:,0])
    # We pick out the customer features and pass them into the customer model.
    customer_embeddings = self.customer_model(features[:,0])
    print('customer_embedding looks like: ', customer_embeddings)
    # And pick out the article features and pass them into the article model,
    # getting embeddings back.
    positive_article_embeddings = self.article_model(features[:,1])
    print('postitive_article_embedding looks like: ', positive_article_embeddings)

    # The task computes the loss and the metrics.

    return self.task(customer_embeddings, positive_article_embeddings, compute_metrics=not training)




splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]

model = RetrievalModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# wE need to create tensor 
train_ds = tf.data.Dataset.from_tensor_slices(dict(train[['customer_id','article_id']]))
print(train_ds.shape())
train_ds_v2=tf.convert_to_tensor(train[['customer_id','article_id']])
dataVar_tensor = tf.constant(train[['customer_id','article_id']], shape=train[['customer_id','article_id']].shape)
test_ds = tf.data.Dataset.from_tensor_slices(dict(test[['customer_id','article_id']]))
num_epochs = 3

model.fit(train_ds_v2, epochs=num_epochs)


model.evaluate(test_ds, return_dict=True)


scann = tfrs.layers.factorized_top_k.ScaNN(model.customer_model, k=5)
scann.index_from_dataset(
    tf.data.Dataset.zip((articles, articles.map(model.article_model)))
)