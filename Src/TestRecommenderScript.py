import numpy as np
import pandas as pd
import tensorflow as tf
from WorkshopExample import SimpleRecommender

test_sub = pd.read_csv('Data/Preprocessed/TestData.csv')
articles = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
articles_sub = articles[['article_id']].values.flatten()
customers_sub = customers[['customer_id']].values.flatten()

test_customer = '6f494dbbc7c70c04997b14d3057edd33a3fc8c0299362967910e80b01254c656'
test_article = 806388002


# Create a new model instance
model = SimpleRecommender(customers_sub, articles_sub, 62)

# Load the previously saved weights
latest = tf.train.latest_checkpoint('Models/')
model.load_weights(latest)

print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

print("Recs for item {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
