import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from Src.BNS import BNS
from Src.deepFM import DeepFactorizationMachineModel
from Src.Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron
from Src.CreateNegativeSamples import CreateNegativeSamples

def sigmoid(x):
    if x > 0 :
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x) /(1+np.exp(x))

class CreateDataset(Dataset):
    def __init__(self, dataset):#, features, idx_variable):

        self.dataset = dataset[:,0:-1]
        self.targets = dataset[:,-1:].float()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, row):
        return self.dataset[row], self.targets[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

train_df = pd.read_csv('Data/Preprocessed/train_df.csv')[0:100000]
valid_df = pd.read_csv('Data/Preprocessed/valid_df.csv')
test_df = pd.read_csv('Data/Preprocessed/test_with_negative_subset.csv')

with open(r"Data/Preprocessed/number_uniques_dict.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

article_df = pd.read_csv('Data/Preprocessed/article_df_numeric.csv')
customer_df = pd.read_csv('Data/Preprocessed/customer_df_numeric.csv')

u_list = customer_df['customer_id']
i_list = article_df['article_id']

i_list2 = train_df['article_id'].unique()
u_list2 = train_df['customer_id'].unique()

u_count = number_uniques_dict['n_customers']
i_count = number_uniques_dict['n_products']

train_df = pd.read_csv('Data/Preprocessed/train_df.csv')[0:600000]
train_tensor = torch.tensor(train_df.fillna(0).to_numpy(), dtype = torch.int)
train_dataset = CreateDataset(train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

batch_size = 1024

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

trainBNS_tensor = torch.zeros(u_count, i_count)
PATH = 'Models/DeepFM_model.pth'
DeepFMModel = torch.load(PATH)

DeepFMModel.eval()
for batch, (X,y) in enumerate(train_loader):
    # Make predictions for this batch
    outputs = DeepFMModel(X)
    trainBNS_tensor[X[:,0],X[:,1]] = outputs

