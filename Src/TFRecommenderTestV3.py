import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import numpy as np
import pprint
from typing import Dict, Text

#CUDA_VISIBLE_DEVICES=""

ratings = tfds.load("movielens/100k-ratings", split = "train")
movies = tfds.load("movielens/100k-movies", split="train")


### Se data:
for x in movies.take(3).as_numpy_iterator():
    pprint.pprint(x)

for x in ratings.take(3).as_numpy_iterator():
    pprint.pprint(x)

tf.random.set_seed(42)

shuffled = ratings.shuffle(100_100, seed = 42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)#.map(lambda x: tf.gather(x['movie_title'], [0], axis = 0))
user_ids = ratings.batch(1_000_000).map(lambda x: x['user_id'])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))

unique_user_id = np.unique(np.concatenate(list(user_ids)))

embedding_dimension = 64

## Se unique movies:
unique_movie_titles[:4]

user_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary = unique_user_id, mask_token = None),
    tf.keras.layers.Embedding(len(unique_user_id)+1,embedding_dimension)
])

movie_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary = unique_movie_titles, mask_token = None),
    tf.keras.layers.Embedding(len(unique_movie_titles)+1,embedding_dimension)
])

metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(movie_model),
    k = 100
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)

class MovielensModel(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task
    
    def compute_loss(self, features: Dict[Text, tf.tensor], training= False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        psoitive_movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, psoitive_movie_embeddings)
        # return super().compute_loss(inputs, training) 

model = MovielensModel(user_model, movie_model)
model.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cahce()
cahced_test = test.batch(4096).cache()