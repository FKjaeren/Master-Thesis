from turtle import color
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from Src.BNS import BNS
#from Src.deepFM import DeepFactorizationMachineModel
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

train_df = pd.read_csv('Data/Preprocessed/train_df.csv')[0:10000]
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

"""
u_count = number_uniques_dict['n_customers']
i_count = number_uniques_dict['n_products']

u_list = pd.read_csv('Data/Preprocessed/customer_df_numeric.csv')['customer_id']
i_list = pd.read_csv('Data/Preprocessed/article_df_numeric.csv')['article_id']
### Udkast til hvordan det kan gøres med ALLE produckter
popularity = np.zeros(i_count-1)

popularity_temp = train_df[['article_id']].value_counts().sort_index().to_frame().reset_index().rename({'article_id':'article_id',0:'counts'},axis = 1)
popularity_temp = i_list.merge(popularity_temp, how = 'left', on = 'article_id').fillna(0)

popularity[:] = popularity_temp['counts']


dict_negative_items = {}
for u in u_list:
    positive_items = set(train_df[train_df['customer_id']==u]['article_id'])
    negative_items = set(i_list)-positive_items
    dict_negative_items[u] = list(negative_items)
"""

popularity = np.zeros(i_count-1)
popularity_temp = pd.read_csv('Data/Preprocessed/article_df_numeric.csv')[['article_id']].value_counts().sort_index().to_frame().reset_index().rename({'article_id':'article_id',0:'counts'},axis = 1)
popularity_temp = pd.DataFrame(i_list).rename({0:'article_id'},axis=1).merge(popularity_temp, how = 'left', on = 'article_id').fillna(0)

popularity[:] = popularity_temp['counts']

prior = popularity/sum(popularity)
"""
interactions_list = []
for c in u_list2:
    for item in i_list2:
        interactions_list.append([c,item,0])

negative_df = pd.DataFrame(data = interactions_list, columns = ['customer_id','article_id','negative_values'])
train_with_negative = train_df.merge(negative_df, how = 'outer', on = ['customer_id','article_id']).fillna(0).drop('negative_values',axis=1)
"""
PATH = 'Models/DeepFM_model.pth'
model = torch.load(PATH)

### Construct training data as we know it:
train_tensor = torch.tensor(train_df.fillna(0).to_numpy(), dtype = torch.int)
valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)
test_tensor = torch.tensor(test_df.fillna(0).to_numpy(), dtype = torch.int)

train_dataset = CreateDataset(train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
valid_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
test_dataset = CreateDataset(test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

batch_size = 1024

dataset_shapes = {'train_shape':train_tensor.shape,'valid_shape':valid_tensor.shape,'test_shape':test_tensor.shape}


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

lr = 0.01
reg = 0.01
d = 1

miu = 0
sigma = reg ** 0.5

U = np.random.normal(loc=miu,
                         scale=sigma,
                         size=[u_count, d])
V = np.random.normal(loc=miu,
                         scale=sigma,
                         size=[i_count, d])

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(95.0))
negative_samples = 50
for batch, (X,y) in enumerate(train_loader):
    outputs = torch.zeros(batch_size,negative_samples+1)
    negative_df = CreateNegativeSamples(None, X, num_negative_samples=negative_samples, type_df=None, method = 'OneCustomerNegSamples', customer_id=X[:,0], article_df=article_df, 
    customer_df=customer_df, batch_size=batch_size)
    data = torch.cat((torch.cat((X,y),dim=1).view(batch_size,1,21),negative_df), dim=1)
    data = data.int()
    X_proc =  data[:,:,:20]
    y_proc = data[:,:,20]
    score_model = torch.zeros(u_count,i_count)
    score_bayesian = torch.zeros(u_count, i_count)
    for i in range(negative_samples+1):
        output = model(X_proc[:,i,:])
        outputs[:,i] = output
    for (x, y_it,output) in zip(X_proc,y_proc,outputs.view(batch_size,negative_samples+1,1)):
        outputs_proc = torch.zeros([i_count])
        y_proc = torch.zeros([i_count]).int()
        for i_idx in range(negative_samples+1):
            output = output.squeeze()
            u = x[:,0].unique()
            i = x[i_idx,1]
            y_proc[i.numpy()] = y_it[i_idx]
            neg_idx = ((y_it == 0).nonzero(as_tuple=True)[0])
            pos_idx = ((y_it == 1).nonzero(as_tuple=True)[0])
            negative_items = x[neg_idx][:,1]
            positive_items = x[pos_idx][:,1]
            outputs_proc[i.numpy()] = output[i_idx]
            rating_vector = np.array(np.mat(U[int(u)]) * np.mat(V.T))[0]
            ################### STARTING NEGATIVE SAMPLING ################### 
            size = 5
            alpha = 5
            #j = BNS(positive_items,negative_items, outputs_proc,prior, size, alpha)
            j = BNS(positive_items,negative_items, rating_vector,prior, size, alpha)
            #r_uij = outputs_proc[i.detach().numpy()] - outputs_proc[j]
            r_uij = rating_vector[i.detach().numpy()] - rating_vector[j]
            # update U and V
            loss_func = 1 - torch.sigmoid(r_uij)
            # update U and V
            U[u] += (lr * (loss_func.detach() * (V[i_idx] - V[j]) - reg * U[u])).item()
            V[i] += (lr * (loss_func.detach() * U[u] - reg * V[i])).item()
            V[j] += (lr * (loss_func.detach() * (-U[u]) - reg * V[j])).item()
        score_model[u,:] = outputs_proc
        score_model[u,positive_items] = -100
    score_bayesian = np.array(np.mat(U) * np.mat(V.T))
    score_bayesian[X[:,0],X[:,1]] = -100


plt.hist(U, bins = 100)
plt.hist(V, bins = 100)
plt.show()

score_bayesian_df = pd.DataFrame(score_bayesian,index=False)
score_bayesian_df.to_csv('Results/bayesian_Scores.csv', index = False)