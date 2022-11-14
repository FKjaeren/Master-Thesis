import numpy as np
import pandas as pd
import tensorflow as tf

data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
#transactions_df = pd.read_csv(data_path+'transactions_train.csv')
transactions_df = pd.read_csv('Data/Preprocessed/transactions_df_subset.csv')


splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]

#df_a_sub = df_a.drop([])
articles_sub = articles_df[['article_id']].values.flatten()
customers_sub = customer_df[['customer_id']].values.flatten()

### Create data with all features:

class SimpleRecommender(tf.keras.Model):
    def __init__(self, customers_sub, articles_sub, embedding_dim):
        super(SimpleRecommender, self).__init__()
        articles_sub = articles_sub['article_id'].unique().astype(str)
        customers_sub = customers_sub['customer_id'].unique().astype(str)
        
        self.articles = tf.constant(articles_sub, dtype=tf.int32)
        self.customers = tf.constant(customers_sub, dtype=tf.string)

        #self.article_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(self.articles, range(len(articles_sub))),-1)
        self.customer_table = tf.keras.layers.StringLookup(vocabulary=customers_sub, mask_token=None, output_mode='int')

        self.article_table = tf.keras.layers.StringLookup(vocabulary=articles_sub, mask_token=None, output_mode='int')
        #self.customer_embed = tf.keras.layers.Embedding(len(customers_data),embedding_dim,customers_data.shape[1])

        #self.articles_embed = tf.keras.layers.Embedding(len(articles_data),embedding_dim, articles_data.shape[1])
        self.dot = tf.keras.layers.Dot(axes=-1)

    def call(self, inputs):
        user = inputs[0]
        article = inputs[1]
        customer_embedding_index = self.customer_table(users['customer_id'].astype(str))
        article_embedding_index = self.article_table(features['article_id'])
        customer_data['customer_id'] = customer_embedding_index.numpy()
        article_data = article
            article_data['article_id'] = article_embedding_index.numpy()
        #customer_embbeding_values = self.customer_embed(customer_embedding_index)
        #article_embedding_values = self.articles_embed(article_embedding_index)

        #return tf.squeeze(self.dot([customer_embbeding_values, article_embedding_values]),1)
        return tf.squeeze(self.dot([customer_data, article_data]),1)
    
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
    #article_tensor = tf.constant(df.drop(['customer_id','t_dat','prod_name','department_name','colour_group_name'],axis=1).values,dtype=tf.int32)
    article_tensor = tf.constant(df[['article_id']].values,dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    dataset = dataset.map(Mapper(articles, number_negative_articles)) 
    dataset = dataset.batch(1024)
    return dataset 

for (customer, candidate), y in get_dataset(train,articles_sub,4):
    print(customer)
    print(candidate)
    print(y)
    break

model = SimpleRecommender(customer_df, articles_df, 64)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.SGD(learning_rate = 100.), 
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(get_dataset(train, articles_sub, 100), validation_data = get_dataset(valid, articles_sub,100), epochs =5)
model.fit(train, validation_data = valid, epochs =5)

test_customer = '6f494dbbc7c70c04997b14d3057edd33a3fc8c0299362967910e80b01254c656'
test_article = 806388002

print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

print("Recs for customer {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
