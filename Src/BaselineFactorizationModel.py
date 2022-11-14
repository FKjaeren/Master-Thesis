import numpy as np
import pandas as pd
import tensorflow as tf


train = pd.read_csv('Data/Preprocessed/train_df_subset.csv')
valid = pd.read_csv('Data/Preprocessed/valid_df_subset.csv')


train_sub = train[['customer_id','article_id']]
valid_sub = valid[['customer_id','article_id']]

articles = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv')
customers = pd.read_csv('Data/Preprocessed/customer_df_numeric_subset.csv')
articles_sub = articles[['article_id']].values.flatten()
customers_sub = customers[['customer_id']].values.flatten()


class SimpleRecommender(tf.keras.Model):
    def __init__(self, customers_sub, articles_sub, embedding_dim):
        super(SimpleRecommender, self).__init__()
        self.articles = tf.constant(articles_sub, dtype=tf.int32)
        self.customers = tf.constant(customers_sub, dtype=tf.string)

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
    
    def call_item_item(self, article):
        article_x = self.article_table.lookup(article)
        article_embeddings = tf.expand_dims(self.articles_embed(article_x),0)
        all_articles_embeddings = tf.expand_dims(self.articles_embed.embeddings,0)
        scores = tf.reshape(self.dot([article_embeddings, all_articles_embeddings]), [-1])

        top_scores, top_indeces = tf.math.top_k(scores, k = 100)
        top_ids = tf.gather(self.articles, top_indeces)
        return top_ids, top_scores
    def Customer_recommendation(self, customer, k):
        customer_x = self.customer_table.lookup(customer)
        customer_embeddings = tf.expand_dims(self.customer_embed(customer_x),0)
        all_articles_embeddings = tf.expand_dims(self.articles_embed.embeddings,0)
        scores = tf.reshape(self.dot([customer_embeddings, all_articles_embeddings]), [-1])

        top_scores, top_indeces = tf.math.top_k(scores, k = k)
        top_ids = tf.gather(self.articles, top_indeces)
        return top_ids, top_scores


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


def get_dataset(df, articles, number_negative_articles):
    dummy_customer_tensor = tf.constant(df[['customer_id']].values, dtype =tf.string)
    article_tensor = tf.constant(df[['article_id']].values,dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    dataset = dataset.map(Mapper(articles, number_negative_articles)) 
    dataset = dataset.batch(1024)
    return dataset 

### Train model

model = SimpleRecommender(customers_sub, articles_sub, 15)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.SGD(learning_rate = 100.), 
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(get_dataset(train_sub, articles_sub, 100), validation_data = get_dataset(valid_sub, articles_sub,100), epochs =5)

# Path
path = 'Models/BaselineModelIteration2'

## save
#model.save_weights(path)

#Forsøg 2 på at save

tf.saved_model.save(model, path)