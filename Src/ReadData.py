import pandas as pd
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import pickle
import os

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

def ReadData(product, customer, features, batch_size, Subset = False):
    prod_features= copy.deepcopy(features)
    customer_features = copy.deepcopy(features)
    customer_features.insert(0, customer)
    prod_features.insert(0, product)
    if(Subset == True):
        UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts.csv')[prod_features]

        Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame.csv')[customer_features][0:150000]
        Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame.csv')[prod_features][0:150000]

        if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
            print('There is dimesion error in the data used for the feed forward (model input)')

        splitrange = round(0.75*len(Customer_Preprocessed_data['customer_id']))
        splitrange2 = round(0.95*len(Customer_Preprocessed_data['customer_id']))
        train_customer = Customer_Preprocessed_data.iloc[:splitrange]
        valid_customer = Customer_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_customer = Customer_Preprocessed_data.iloc[splitrange2:]

        train_product = Product_Preprocessed_data.iloc[:splitrange]
        valid_product = Product_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_product = Product_Preprocessed_data.iloc[splitrange2:]
    else:
        UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts.csv')[prod_features]

        Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame.csv')[customer_features]
        Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame.csv')[prod_features]

        if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
            print('There is dimesion error in the data used for the feed forward (model input)')

        splitrange = round(0.75*len(Customer_Preprocessed_data['customer_id']))
        splitrange2 = round(0.95*len(Customer_Preprocessed_data['customer_id']))
        train_customer = Customer_Preprocessed_data.iloc[:splitrange]
        valid_customer = Customer_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_customer = Customer_Preprocessed_data.iloc[splitrange2:]

        train_product = Product_Preprocessed_data.iloc[:splitrange]
        valid_product = Product_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_product = Product_Preprocessed_data.iloc[splitrange2:]

    with open(r"Data/Preprocessed/number_uniques_dict.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)

    #Customer_data_tensor = torch.tensor(Only_Customer_data[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
    Product_data_tensor = torch.tensor(UniqueProducts_df.fillna(0).to_numpy(), dtype = torch.int)
    customer_train_tensor = torch.tensor(train_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_train_tensor = torch.tensor(train_product.fillna(0).to_numpy(), dtype = torch.int)

    customer_valid_tensor = torch.tensor(valid_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_valid_tensor = torch.tensor(valid_product.fillna(0).to_numpy(), dtype = torch.int)
    #customer_dataset = CreateDataset(Customer_data_tensor)#,features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_dataset = CreateDataset(Product_data_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])

    customer_test_tensor = torch.tensor(test_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_test_tensor = torch.tensor(test_product.fillna(0).to_numpy(), dtype = torch.int)

    product_train_dataset = CreateDataset(product_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_train_dataset = CreateDataset(customer_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_valid_dataset = CreateDataset(product_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_valid_dataset = CreateDataset(customer_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_test_dataset = CreateDataset(product_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_test_dataset = CreateDataset(customer_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

    dataset_shapes = {'train_shape':product_train_tensor.shape,'valid_shape':product_valid_tensor.shape,'test_shape':product_test_tensor.shape}

    #Training in batches:
    product_train_loader = torch.utils.data.DataLoader(product_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)
    customer_train_loader = torch.utils.data.DataLoader(customer_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    product_valid_loader = torch.utils.data.DataLoader(product_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
    customer_valid_loader = torch.utils.data.DataLoader(customer_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    product_test_loader = torch.utils.data.DataLoader(product_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
    customer_test_loader = torch.utils.data.DataLoader(customer_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    return product_dataset, product_train_loader, customer_train_loader, product_valid_loader, customer_valid_loader, number_uniques_dict, dataset_shapes, product_test_loader, customer_test_loader
