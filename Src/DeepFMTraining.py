import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
import copy
from Src.Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron#, FeaturesLinear

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


train_df = pd.read_csv('Data/Preprocessed/train_df.csv')[0:600000]
valid_df = pd.read_csv('Data/Preprocessed/valid_df.csv')
test_df = pd.read_csv('Data/Preprocessed/test_df.csv')

device = torch.device('cpu')

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

with open(r"Data/Preprocessed/number_uniques_dict.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A Pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, n_unique_dict, device, batch_size):
        super().__init__()
        #self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(embedding_dim = embed_dim,num_fields=field_dims ,batch_size= batch_size, n_unique_dict=n_unique_dict, device = device)
        self.embed_output_dim = (len(field_dims)-1) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        if(torch.isnan(embed_x).sum() > 0):
            print("Values with nan in embedding output: ",embed_x[torch.isnan(embed_x)])
        if(torch.isnan(self.fm(embed_x)).sum() > 0):
            print("Values with nan in fm output: ",self.fm(embed_x)[torch.isnan(self.fm(embed_x))])
        if(torch.isnan(self.mlp(embed_x.view(-1, self.embed_output_dim))).sum() > 0):
            print("Values with nan in mlp output: ",self.mlp(embed_x.view(-1, self.embed_output_dim))[torch.isnan(self.mlp(embed_x.view(-1, self.embed_output_dim)))])
        #x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        #x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        if(torch.isnan(torch.sigmoid(x.squeeze(1))).sum() > 0):
            print("Values with nan in sigmoid output: ",torch.sigmoid(x.squeeze(1))[torch.isnan(torch.sigmoid(x.squeeze(1)))])
        return torch.sigmoid(x.squeeze(1))
    def Reccomend_topk(x, k):
        item_idx = torch.topk(x,k)
        return item_idx


embedding_dim = 16
DeepFMModel = DeepFactorizationMachineModel(field_dims = train_df.columns, embed_dim=embedding_dim, mlp_dims=[16,32,16], dropout=0.2, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size)
optimizer = torch.optim.Adam(DeepFMModel.parameters(), weight_decay=0.00001, lr = 0.002)
pos_weight = 0.95
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))

num_epochs = 1

Loss_list = []
Valid_Loss_list = []
Best_loss = np.infty
for epoch in range(1,num_epochs+1):
    print(epoch)
    running_loss = 0.
    epoch_loss = []
    DeepFMModel.train()

    for batch, (X,y) in enumerate(train_loader):
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #
        dataset = X.to(device)
        # Make predictions for this batch
        outputs = DeepFMModel(dataset)
        if(torch.isnan(X).sum() > 0):
            print("SE her Values with nan in X: ",X[torch.isnan(X)])
        if(torch.isnan(outputs).sum() > 0):
            print("Values with nan in outputs: ",outputs[torch.isnan(outputs)])
            print("And the batch is: ", batch)
        # Compute the loss and its gradients

        loss = loss_fn(outputs,y.squeeze().to(device))
        loss.backward()
        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        epoch_loss.append(loss.item())
        if(batch % 100 == 0):
            print(' Train batch {} loss: {}'.format(batch, np.mean(epoch_loss)))

    if(epoch % 1 == 0):
        print(' Train epoch {} loss: {}'.format(epoch, np.mean(epoch_loss)))
        
        epoch_loss.append(loss.item())
    epoch_loss_value = np.mean(epoch_loss)
    Loss_list.append(epoch_loss_value)
    DeepFMModel.eval()
    epoch_valid_loss = []
    for batch, (X_valid,y_valid) in enumerate(valid_loader):
        outputs = DeepFMModel(X_valid)
        loss = loss_fn(outputs,y_valid.squeeze())
        epoch_valid_loss.append(loss.item())
    if(epoch % 1 == 0):
        print(' Valid epoch {} loss: {}'.format(epoch, np.mean(epoch_valid_loss)))
    epoch_valid_loss_value = np.mean(epoch_valid_loss)
    Valid_Loss_list.append(epoch_valid_loss_value)
    if(epoch_valid_loss_value < Best_loss):
        best_model = copy.deepcopy(DeepFMModel)
        Best_loss = epoch_valid_loss_value
PATH = 'Models/DeepFM_model.pth'
torch.save(best_model, PATH)

print("finished training")
print("Loss list = ", Loss_list)
##test:
# 1 iteration:

dataiter = iter(test_loader)
X, y = next(dataiter)

output = DeepFMModel(X)

predictions = []
for i in output:
    if i < 0.5:
        predictions.append(0)
    else:
        predictions.append(1)

print("The output of the model is: ", predictions)

print("The true labels are: ", y)

print("The accuracy of the model on 1 iterations is:", (1-(torch.sum(y.squeeze() - torch.tensor(predictions, dtype = torch.int)).item())/len(y))*100,"%")

