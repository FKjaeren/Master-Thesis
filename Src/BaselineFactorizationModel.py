import numpy as np
import pandas as pd
import tensorflow as tf

# Load data for lookup tables and split in training and validation
df = pd.read_csv('Data/Raw/transactions_train_subset.csv')
splitrange = round(0.8*len(df['customer_id']))
splitrange2 = round(0.975*len(df['customer_id']))
train = df.iloc[:splitrange]
valid = df.iloc[splitrange+1:splitrange2]


train_sub = train[['customer_id','article_id']]
valid_sub = valid[['customer_id','article_id']]

articles = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv')
customers = pd.read_csv('Data/Preprocessed/customer_df_numeric_subset.csv')
customer_raw = pd.read_csv('Data/Raw/customers_subset.csv')
articles_raw = pd.read_csv('Data/Raw/articles_subset.csv')
articles_raw = articles_raw[['article_id']].values.flatten()
customer_raw = customer_raw[['customer_id']].values.flatten()

# Baseline model with two fetaures and how to find top k
class SimpleRecommender(tf.keras.Model):
    def __init__(self, customers_sub, articles_sub, embedding_dim):
        super(SimpleRecommender, self).__init__()
        self.articles = tf.constant(articles_raw, dtype=tf.int32)
        self.customers = tf.constant(customer_raw, dtype=tf.string)

        # Lookup tables for article id and customer id
        self.article_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.articles, range(len(articles_sub))),-1)
        self.customer_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.customers, range(len(customers_sub))),-1)

        self.customer_embed = tf.keras.layers.Embedding(len(customers_sub),embedding_dim)

        self.articles_embed = tf.keras.layers.Embedding(len(articles_sub),embedding_dim)

        self.dot = tf.keras.layers.Dot(axes=-1)

    def call(self, inputs):
        user = inputs[0]
        article = inputs[1]

        customer_embedding_index = self.customer_table.lookup(user)
        article_embedding_index = self.article_table.lookup(article)

        customer_embbeding_values = self.customer_embed(customer_embedding_index)
        article_embedding_values = self.articles_embed(article_embedding_index)

        return tf.squeeze(self.dot([customer_embbeding_values, article_embedding_values]),1)
    # Get recommendatios for customers
    def _Customer_recommendation(self, customer, k):
        customer_x = self.customer_table.lookup(customer)
        customer_embeddings = tf.expand_dims(self.customer_embed(customer_x),0)
        all_articles_embeddings = tf.expand_dims(self.articles_embed.embeddings,0)
        scores = tf.reshape(self.dot([customer_embeddings, all_articles_embeddings]), [-1])
        print(scores.shape)
        top_scores, top_indeces = tf.math.top_k(scores, k = k)
        top_ids = tf.gather(self.articles, top_indeces)
        return top_ids, top_scores
    
    @tf.function(
        input_signature=[tf.TensorSpec(shape=(1, 1), dtype=tf.string), tf.TensorSpec(shape=(None), dtype=tf.int32)]
    )
    def Customer_recommendation(self, customer, k):
        return self._Customer_recommendation(customer, k)

# Mapper function which also makes negative samples with a one hot encoding
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

# Get dataset in batches
def get_dataset(df, articles, number_negative_articles):
    dummy_customer_tensor = tf.constant(df[['customer_id']].values, dtype =tf.string)
    article_tensor = tf.constant(df[['article_id']].values,dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    dataset = dataset.map(Mapper(articles, number_negative_articles)) 
    dataset = dataset.batch(256)
    return dataset 

### Train model

model = SimpleRecommender(customer_raw, articles_raw, 32)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.SGD(learning_rate = 0.001), 
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(get_dataset(train_sub, articles_raw, 10), validation_data = get_dataset(valid_sub, articles_raw, 10), epochs =5, verbose=1)

# Path
path = 'Models/test_baseline_model_with_tf_function2'

## save
#model.save_weights(path, save_format='tf')
#Forsøg 2 på at save
model.save(path, save_format='tf')
#forsøg 3
#tf.saved_model.save(model, path)