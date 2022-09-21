import numpy as np
import pandas as pd
import tensorflow as tf
import torch
#from WorkshopExample import SimpleRecommender
#from PytorchTestV3 import RecSysModel
#from PytorchTestV3 import CreateDataset
from torch.utils.data import Dataset

PATH = 'Models/Baseline_MulitDim_model.pth'

class CreateDataset(Dataset):
    def __init__(self, dataset, features, idx_variable):

        self.id = idx_variable
        self.features = features
        self.all_data = dataset

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, row):

        features = torch.tensor(self.all_data[self.features].to_numpy(), dtype = torch.int)
        idx_variable = torch.tensor(self.all_data[self.id].to_numpy(), dtype = torch.int)
        all_data = torch.cat((idx_variable, features), dim = 1)
        return all_data[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

model = (torch.load(PATH))
all_products = pd.read_csv('Data/Raw/articles.csv')
num_products = all_products['prod_name'].nunique()


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

batch_size = 1024

test_dataset = pd.read_csv('Data/Preprocessed/test_final.csv')
product_test_dataset = CreateDataset(test_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])
customer_test_dataset = CreateDataset(test_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

product_test_loader = torch.utils.data.DataLoader(product_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)
customer_test_loader = torch.utils.data.DataLoader(customer_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
total_accuracy = []
model.eval()
label_products_init = torch.zeros(num_products, batch_size)
for i, product_data_batch,customer_data_batch in zip(np.arange(1,product_test_dataset.shape()[0]),product_test_loader,customer_test_loader):
    product_id = product_data_batch[:,0].type(torch.long)
    label_products_init[product_id] = 1
    outputs = model.CustomerItemRecommendation(customer_data_batch)
    print(outputs)
    output = torch.squeeze(outputs, 1)
    accuracy = np.sum(np.abs(output-label_products_init))
    total_accuracy.append(accuracy)
    break




print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

print("Recs for item {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
