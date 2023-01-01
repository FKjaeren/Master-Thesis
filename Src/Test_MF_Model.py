
import numpy as np
import pandas as pd
#from Src.BaselineFactorizationModel import SimpleRecommender
from torch.utils.data import Dataset
import copy
import torch
import pickle
from Src.MF_Model_Arch import RecSysModel
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
batch_size = 1

product='article_id'
customer='customer_id'
features= ['club_member_status','fashion_news_frequency', 'age', 'postal_code', 'price',
            'sales_channel_id', 'season', 'day', 'month', 'year', 'prod_name',
            'graphical_appearance_name', 'colour_group_name','department_name']

prod_features= copy.deepcopy(features)
customer_features = copy.deepcopy(features)
customer_features.insert(0, customer)
prod_features.insert(0, product)

## Load data

UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts_subset.csv')[prod_features]

Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame_subset.csv')[customer_features]
Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame_subset.csv')[prod_features]

if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
    print('There is dimesion error in the data used for the feed forward (model input)')

splitrange = round(0.8*len(Customer_Preprocessed_data['customer_id']))
splitrange2 = round(0.975*len(Customer_Preprocessed_data['customer_id']))

# Get the customers which have also been tested on for the other models

test_customer = Customer_Preprocessed_data.iloc[splitrange2+1:]

test_product = Product_Preprocessed_data.iloc[splitrange2+1:]

customers = test_customer.customer_id.unique()[0:12000]

final_idx = test_customer.index[test_customer['customer_id'].isin(customers)].tolist()

test_customer = test_customer.loc[final_idx]
test_product = test_product.loc[final_idx]

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

## Convert data to tensor

Product_data_tensor = torch.tensor(UniqueProducts_df.fillna(0).to_numpy(), dtype = torch.int)

product_dataset = CreateDataset(Product_data_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])

customer_test_tensor = torch.tensor(test_customer.fillna(0).to_numpy(), dtype = torch.int)
product_test_tensor = torch.tensor(test_product.fillna(0).to_numpy(), dtype = torch.int)

embedding_dim = 32
device = "cpu"

#Define model

model = RecSysModel(product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_unique_dict=number_uniques_dict, device=device, n_ages = 111)
path = 'Models/Baseline_MulitDim_model.pth'
model.load_state_dict(torch.load(path))

num_recommendations= 12

one_accuracy_all = []
twelve_accuracy_all = []

#Calculate mAP(1) and mAP(12) for the 12000 customers defined as out testset

for c in customers:
    temp_accuracy = []
    transaction_indexes = test_customer.index[test_customer["customer_id"] == c]
    test_df_temp = test_customer.loc[transaction_indexes].iloc[0]
    true_values = test_product.loc[transaction_indexes].article_id.unique()
    test_df_tensor = torch.tensor(test_df_temp.fillna(0).to_numpy(), dtype = torch.int).unsqueeze(dim = 0)
    output_matrix = model(test_df_tensor,None)
    recommendations, indexes = torch.topk(output_matrix,num_recommendations)
    indexes = indexes.squeeze()
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
    if(num_recommendations <= len(true_values)):
        temp_accuracy = sum(temp_accuracy)/num_recommendations
    else:
        temp_accuracy = sum((np.sort(temp_accuracy)[::-1][0:len(true_values)]))/len(true_values)

    twelve_accuracy_all.append(temp_accuracy)

one_accuracy_all_final = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all_final = sum(twelve_accuracy_all)/len(customers)





print("The accuracy at hitting one correct recommendation is: ",one_accuracy_all_final*100, "%")
print("The accuracy at hitting 12 accurate recommendations is ",twelve_accuracy_all_final*100,"%")
print(f"The model was tested on {len(customers)} users")