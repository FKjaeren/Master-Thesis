import numpy as np
import pandas as pd
import tensorflow as tf

data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')

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
#datetime and create a month column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month
transactions_df['year'] =  pd.DatetimeIndex(transactions_df['t_dat']).year
transactions_df['day'] =  pd.DatetimeIndex(transactions_df['t_dat']).day



transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 


X = transactions_df.iloc[0:20000].merge(df_a, how = "left", on = "article_id")

X = pd.get_dummies(X, columns = ['season','product_type_name','product_group_name','section_name'])

X = X.merge(df_c[['customer_id','age']], how = 'left', on = 'customer_id')

splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))

train = X.iloc[:splitrange]
valid = X.iloc[splitrange+1:splitrange2]
test = X.iloc[splitrange2:]

#df_a_sub = df_a.drop([])
articles_sub = articles_df[['article_id']].values.flatten()
customers_sub = customer_df[['customer_id']].values.flatten()

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

for (customer, candidate), y in get_dataset(train,articles_sub,4):
    print(customer)
    print(candidate)
    print(y)
    break

model = SimpleRecommender(customers_sub, articles_sub, 15)
model.compile(loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
            optimizer=tf.keras.optimizers.SGD(learning_rate = 100.), 
            metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(get_dataset(train, articles_sub, 100), validation_data = get_dataset(valid, articles_sub,100), epochs =5)

test_customer = '6f494dbbc7c70c04997b14d3057edd33a3fc8c0299362967910e80b01254c656'
test_article = 806388002

print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

print("Recs for item {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
