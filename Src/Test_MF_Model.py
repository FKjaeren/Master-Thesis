
import numpy as np
import pandas as pd
import tensorflow as tf
#from Src.BaselineFactorizationModel import SimpleRecommender
from torch.utils.data import Dataset, TensorDataset
import copy
import torch
import pickle
from MF_Model_Arch import RecSysModel
class CreateDataset(Dataset):
    def __init__(self, dataset):#, features, idx_variable):

        self.all_data = dataset

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, row):

        return self.all_data[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value
batch_size = 256

product='article_id'
customer='customer_id'
features= ['club_member_status','fashion_news_frequency', 'age', 'postal_code', 'price',
            'sales_channel_id', 'season', 'day', 'month', 'year', 'prod_name',
            'graphical_appearance_name', 'colour_group_name','department_name']

prod_features= copy.deepcopy(features)
customer_features = copy.deepcopy(features)
customer_features.insert(0, customer)
prod_features.insert(0, product)
UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts_subset.csv')[prod_features]

Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame_subset.csv')[customer_features]
Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame_subset.csv')[prod_features]

if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
    print('There is dimesion error in the data used for the feed forward (model input)')

splitrange = round(0.8*len(Customer_Preprocessed_data['customer_id']))
splitrange2 = round(0.975*len(Customer_Preprocessed_data['customer_id']))

test_customer = Customer_Preprocessed_data.iloc[splitrange2+1:]

test_product = Product_Preprocessed_data.iloc[splitrange2+1:]

customers = test_customer.customer_id.unique()[0:12000]

final_idx = test_customer.index[test_customer['customer_id'].isin(customers)].tolist()

test_customer = test_customer.loc[final_idx]
test_product = test_product.loc[final_idx]

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

#Customer_data_tensor = torch.tensor(Only_Customer_data[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
Product_data_tensor = torch.tensor(UniqueProducts_df.fillna(0).to_numpy(), dtype = torch.int)
)
product_dataset = CreateDataset(Product_data_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])

customer_test_tensor = torch.tensor(test_customer.fillna(0).to_numpy(), dtype = torch.int)
product_test_tensor = torch.tensor(test_product.fillna(0).to_numpy(), dtype = torch.int)

product_test_dataset = CreateDataset(product_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
customer_test_dataset = CreateDataset(customer_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

#dataset_shapes = {'train_shape':product_train_tensor.shape,'valid_shape':product_valid_tensor.shape,'test_shape':product_test_tensor.shape}

product_test_loader = torch.utils.data.DataLoader(product_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
customer_test_loader = torch.utils.data.DataLoader(customer_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

embedding_dim = 32
device = "cpu"

model = RecSysModel(product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_unique_dict=number_uniques_dict, device=device, n_ages = 111)
path = 'Models/Baseline_MulitDim_model.pth'
model.load_state_dict(torch.load(path))

k= 12

one_accuracy_all = []
twelve_accuracy_all = []

for c in customers:
    temp_accuracy = []
    transaction_indexes = test_customer.index[test_customer["customer_id"] == c]
    test_df_temp = test_customer.loc[transaction_indexes][0]
    true_values = test_product.loc[transaction_indexes].article_id.unique()

    output_matrix = model(test_df_temp)
    recommendations, indexes = torch.topk(output_matrix)
    recommendations = UniqueProducts_df.loc[indexes]['article_id']
    if any(x in recommendations for x in true_values):
        accuracy = 1.0
    else:
        accuracy = 0.0
    one_accuracy_all.append(accuracy)
    for i in recommendations:
        if i in true_values:
            temp_accuracy.append(1)
        else:
            temp_accuracy.append(0)
    if(k <= len(true_values)):
        temp_accuracy = sum(temp_accuracy)/k
    else:
        temp_accuracy = sum((np.sort(temp_accuracy)[::-1][0:len(true_values)]))/len(true_values)

    twelve_accuracy_all.append(temp_accuracy)

one_accuracy_all = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all = sum(twelve_accuracy_all)/len(customers)





print("The accuracy at hitting one correct recommendation is: ",one_accuracy_all, "%")
print("The accuracy at hitting 12 accurate recommendations is ",twelve_accuracy_all,"%")