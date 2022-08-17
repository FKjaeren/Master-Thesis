import tensorflow as tf
 
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movie_lens/100k-ratings", split="train", )
# Features of all the available movies.
movies = tfds.load("movie_lens/100k-movies", split="train")


ratings = ratings.map(lambda x: {"movie_title": x["movie_title"],"user_id": x["user_id"],})
movies = movies.map(lambda x: x["movie_title"])


class TwoTowerMovielensModel(tfrs.Model):
  """MovieLens prediction model."""
 
  def __init__(self):
    # The `__init__` method sets up the model architecture.
    super().__init__()
 
    # How large the representation vectors are for inputs: larger vectors make
    # for a more expressive model but may cause over-fitting.
    embedding_dim = 32
    num_unique_users = 1000
    num_unique_movies = 1700
    eval_batch_size = 128

    # Set up user and movie representations.
    self.user_model = tf.keras.Sequential([
      # We first turn the raw user ids into contiguous integers by looking them
      # up in a vocabulary.
      tf.keras.layers.experimental.preprocessing.StringLookup(
          max_tokens=num_unique_users),
      # We then map the result into embedding vectors.
      tf.keras.layers.Embedding(num_unique_users, embedding_dim)
    ])
    
    self.movie_model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
          max_tokens=num_unique_movies),
      tf.keras.layers.Embedding(num_unique_movies, embedding_dim)
    ])
    # The `Task` objects has two purposes: (1) it computes the loss and (2)
    # keeps track of metrics.
    self.task = tfrs.tasks.Retrieval(
        # In this case, our metrics are top-k metrics: given a user and a known
        # watched movie, how highly would the model rank the true movie out of
        # all possible movies?
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(eval_batch_size).map(self.movie_model)
        )
    )
    def compute_loss(self, features, training=False) -> tf.Tensor:
        # The `compute_loss` method determines how loss is computed.
    
        # Compute user and item embeddings.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
    
        # Pass them into the task to get the resulting loss. The lower the loss is, the
        # better the model is at telling apart true watches from watches that did
        # not happen in the training data.
        return self.task(user_embeddings, movie_embeddings)

model = TwoTowerMovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
 
model.fit(ratings.batch(4096), verbose=False)