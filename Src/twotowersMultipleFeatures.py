import numpy as np
import pandas as pd
import tensorflow as tf
import os
from typing import Dict, Text
import pprint
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
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





for x in articles.take(1).as_numpy_iterator():
  pprint.pprint(x)
feature_names = ["customer_id", "prod_name"]
vocabularies = {}

for feature_name in feature_names:
  vocab = interactions.batch(1_000_000).map(lambda x: x[feature_name])
  vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

class DCN(tfrs.Model):

  def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
    super().__init__()

    self.embedding_dimension = 32

    str_features = ["customer_id", "prod_name"]
    #int_features = ["user_gender", "bucketized_user_age"]

    self._all_features = str_features #+ int_features
    self._embeddings = {}

    # Compute embeddings for string features.
    for feature_name in str_features:
      vocabulary = vocabularies[feature_name]
      self._embeddings[feature_name] = tf.keras.Sequential(
          [tf.keras.layers.StringLookup(
              vocabulary=vocabulary, mask_token=None),
           tf.keras.layers.Embedding(len(vocabulary) + 1,
                                     self.embedding_dimension)
    ])

    """ # Compute embeddings for int features.
    for feature_name in int_features:
      vocabulary = vocabularies[feature_name]
      self._embeddings[feature_name] = tf.keras.Sequential(
          [tf.keras.layers.IntegerLookup(
              vocabulary=vocabulary, mask_value=None),
           tf.keras.layers.Embedding(len(vocabulary) + 1,
                                     self.embedding_dimension)
    ])
 """
    if use_cross_layer:
      self._cross_layer = tfrs.layers.dcn.Cross(
          projection_dim=projection_dim,
          kernel_initializer="glorot_uniform")
    else:
      self._cross_layer = None

    self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
      for layer_size in deep_layer_sizes]

    self._logit_layer = tf.keras.layers.Dense(1)

    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
    )

  def call(self, features):
    # Concatenate embeddings
    embeddings = []
    for feature_name in self._all_features:
      embedding_fn = self._embeddings[feature_name]
      embeddings.append(embedding_fn(features[feature_name]))

    x = tf.concat(embeddings, axis=1)

    # Build Cross Network
    if self._cross_layer is not None:
      x = self._cross_layer(x)

    # Build Deep Network
    for deep_layer in self._deep_layers:
      x = deep_layer(x)

    return self._logit_layer(x)

  def compute_loss(self, features, training=False):
    labels = features.pop("quantity")
    scores = self(features)
    return self.task(
        labels=labels,
        predictions=scores,
    )




tf.random.set_seed(42)
shuffled = interactions.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(8192).cache()


cached_test = test.batch(4096).cache()



def run_models(use_cross_layer, deep_layer_sizes, projection_dim=None, num_runs=5):
  models = []
  rmses = []

  for i in range(num_runs):
    model = DCN(use_cross_layer=use_cross_layer,
                deep_layer_sizes=deep_layer_sizes,
                projection_dim=projection_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    models.append(model)

    model.fit(cached_train, epochs=epochs, verbose=False)
    metrics = model.evaluate(cached_test, return_dict=True)
    rmses.append(metrics["RMSE"])

  mean, stdv = np.average(rmses), np.std(rmses)

  return {"model": models, "mean": mean, "stdv": stdv}

epochs = 8
learning_rate = 0.01

dcn_result = run_models(use_cross_layer=True,
                        deep_layer_sizes=[192, 192])



dcn_lr_result = run_models(use_cross_layer=True,
                           projection_dim=20,
                           deep_layer_sizes=[192, 192])

dnn_result = run_models(use_cross_layer=False,
                        deep_layer_sizes=[192, 192, 192])


print("DCN            RMSE mean: {:.4f}, stdv: {:.4f}".format(
    dcn_result["mean"], dcn_result["stdv"]))
print("DCN (low-rank) RMSE mean: {:.4f}, stdv: {:.4f}".format(
    dcn_lr_result["mean"], dcn_lr_result["stdv"]))
print("DNN            RMSE mean: {:.4f}, stdv: {:.4f}".format(
    dnn_result["mean"], dnn_result["stdv"]))



model = dcn_result["model"][0]
mat = model._cross_layer._dense.kernel
features = model._all_features

block_norm = np.ones([len(features), len(features)])

dim = model.embedding_dimension

# Compute the norms of the blocks.
for i in range(len(features)):
  for j in range(len(features)):
    block = mat[i * dim:(i + 1) * dim,
                j * dim:(j + 1) * dim]
    block_norm[i,j] = np.linalg.norm(block, ord="fro")

plt.figure(figsize=(9,9))
im = plt.matshow(block_norm, cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([""] + features, rotation=45, ha="left", fontsize=10)
_ = ax.set_yticklabels([""] + features, fontsize=10)
plt.show()