import numpy as np
import pandas as pd
import tensorflow as tf
import torch
#from WorkshopExample import SimpleRecommender
#from PytorchTestV3 import RecSysModel
#from PytorchTestV3 import CreateDataset
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score
from PytorchTestV4 import ReadData

PATH = 'Models/Baseline_MulitDim_model.pth'


model = (torch.load(PATH))

batch_size = 1024

product_dataset, _, _, _, _, number_uniques_dict, dataset_shapes, product_test_loader , customer_test_loader = ReadData(
                                                            product='article_id', customer='customer_id',features= ['FN', 'Active', 'club_member_status',
                                                            'fashion_news_frequency', 'age', 'postal_code', 'price',
                                                            'sales_channel_id', 'season', 'day', 'month', 'year', 'prod_name',
                                                            'product_type_name', 'graphical_appearance_name', 'colour_group_name',
                                                            'department_name', 'index_group_name'], batch_size=batch_size, Subset= True)


"""
test_sub = pd.read_csv('Data/Preprocessed/TestData.csv')
articles = pd.read_csv('Data/Raw/articles.csv')
customers = pd.read_csv('Data/Raw/customers.csv')
articles_sub = articles[['article_id']].values.flatten()
customers_sub = customers[['customer_id']].values.flatten()

test_customer = '6f494dbbc7c70c04997b14d3057edd33a3fc8c0299362967910e80b01254c656'
test_article = 806388002


# Create a new model instance
model = SimpleRecommender(customers_sub, articles_sub, 15)

# Load the previously saved weights


latest = tf.train.latest_checkpoint('Models/')
model.load_weights(latest)
"""
## Load model forsøg 2 
#path = 'Models/BaselineModelIteration2'
# model = tf.saved_model.load(path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_products = number_uniques_dict['num_article_id']

total_Precision_score = []
model.eval()
label_products_init = torch.zeros(num_products, batch_size)
output_product_init = torch.zeros(num_products, batch_size)
for i, product_data_batch,customer_data_batch in zip(np.arange(1,dataset_shapes['train_shape'][0]),product_test_loader,customer_test_loader):
    product_id = product_data_batch[:,0].type(torch.long)
    label_products_init[product_id] = 1
    test = (torch.nn.functional.one_hot(product_id, num_products))
    values,outputs, probabilities = model.CustomerItemRecommendation(customer_data_batch,1)

    output = torch.squeeze(outputs, 1)
    #output_product_init[(outputs).view(batch_size,1)] = 1

    Precision_score = average_precision_score(test,probabilities.detach().numpy())
    total_Precision_score.append(Precision_score)
    break




#print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

#print("Recs for item {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
