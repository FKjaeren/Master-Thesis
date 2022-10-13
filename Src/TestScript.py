import numpy as np
import pandas as pd
import torch
#from WorkshopExample import SimpleRecommender
#from PytorchTestV3 import RecSysModel
#from PytorchTestV3 import CreateDataset
from sklearn.metrics import average_precision_score
from Src.ReadData import *

device = torch.device("cpu")

PATH = 'Models/Baseline_MulitDim_model.pth'


model = (torch.load(PATH))

batch_size = 1024

product_dataset, _, _, _, _, number_uniques_dict, dataset_shapes, product_test_loader , customer_test_loader = ReadData(
                                                            product='article_id', customer='customer_id',features= ['FN', 'Active', 'club_member_status',
                                                            'fashion_news_frequency', 'age', 'postal_code', 'price',
                                                            'sales_channel_id', 'season', 'day', 'month', 'year', 'prod_name',
                                                            'product_type_name', 'graphical_appearance_name', 'colour_group_name',
                                                            'department_name', 'index_group_name'], batch_size=batch_size, Subset= True)


## Load model forsøg 2 
#path = 'Models/BaselineModelIteration2'
# model = tf.saved_model.load(path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#num_products = number_uniques_dict['num_article_id']

num_products = number_uniques_dict['n_products']
k = 6
model.eval()
batch_top1_accuracy = []
batch_top6_accuracy = []
for i, product_data_batch,customer_data_batch in zip(np.arange(1,dataset_shapes['test_shape'][0]),product_test_loader,customer_test_loader):
    batch_top_1_predictions = []
    batch_top_6_predictions = []
    product_id = product_data_batch[:,0].type(torch.long)
    test = (torch.nn.functional.one_hot(product_id, num_products))
    recommendations,indexes, probabilities = model.CustomerItemRecommendation(customer_data_batch,k)

    for i in range(batch_size):
        if(indexes[i][0] == product_id[i]):
            batch_top_1_predictions.append(1)
        else:
            batch_top_1_predictions.append(0)
        if(product_id[i] in indexes[i]) == True:
            batch_top_6_predictions.append(1)
        else:
            batch_top_6_predictions.append(0)
    batch_top1_accuracy = (sum(batch_top_1_predictions)/batch_size)
    batch_top6_accuracy = (sum(batch_top_6_predictions)/batch_size)

top1_accuracy = np.mean(batch_top1_accuracy)
top6_accuracy = np.mean(batch_top6_accuracy)

print("The model predicts the most likely buy with an accuracy of: ", top1_accuracy*100,"%. And it predicts the 6 most likely buys with an accuracy of", top6_accuracy*100,"%.")



#print("Recs for item {}: {}".format(test_article, model.call_item_item(tf.constant(test_article, dtype=tf.int32))))

#print("Recs for item {}: {}".format(test_customer, model.Customer_recommendation(tf.constant(test_customer, dtype=tf.string), k=12)))
