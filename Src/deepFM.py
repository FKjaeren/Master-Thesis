
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
from Src.Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron#, FeaturesLinear

class CreateDataset(Dataset):
    def __init__(self, dataset, targets):#, features, idx_variable):

        self.dataset = dataset
        self.targets = targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, row):

        return self.dataset[row], self.targets[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

df = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame.csv')[0:130000]

products = pd.read_csv('Data/Preprocessed/FinalProductDataFrame.csv')[0:130000]

product_ids = products[['article_id']]

splitrange = round(0.75*len(df['customer_id']))
splitrange2 = round(0.95*len(df['customer_id']))
train_df = df.iloc[:splitrange]
valid_df = df.iloc[splitrange+1:splitrange2]
test_df = df.iloc[splitrange2:]
device = torch.device("cpu")

train_product_ids = product_ids[:splitrange]
valid_products_ids = product_ids[splitrange+1:splitrange2]

train_tensor_product_ids = torch.tensor(train_product_ids.fillna(0).to_numpy(), dtype = torch.int)
valid_tensor_product_ids = torch.tensor(valid_products_ids.fillna(0).to_numpy(), dtype = torch.int)


train_tensor = torch.tensor(train_df.fillna(0).to_numpy(), dtype = torch.int)
valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)
test_tensor = torch.tensor(test_df.fillna(0).to_numpy(), dtype = torch.int)

train_dataset = CreateDataset(train_tensor,train_tensor_product_ids)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
valid_dataset = CreateDataset(valid_tensor,valid_tensor_product_ids)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
#test_dataset = CreateDataset(test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

batch_size = 1024

dataset_shapes = {'train_shape':train_tensor.shape,'valid_shape':valid_tensor.shape,'test_shape':test_tensor.shape}


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

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
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        #x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        #x = self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))

embedding_dim = 16
DeepFMModel = DeepFactorizationMachineModel(field_dims = df.columns, embed_dim=embedding_dim, mlp_dims=[16,16], dropout=0.2, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size)
optimizer = torch.optim.Adam(DeepFMModel.parameters(), weight_decay=0.00001, lr = 0.005)
loss_fn = torch.nn.BCELoss()

num_epochs = 10

Loss_list = []
Valid_Loss_list = []
Best_loss = np.infty
for epoch in range(1,num_epochs+1):
    print(epoch)
    running_loss = 0.
    epoch_loss = []
    DeepFMModel.train()

    for batch, data in enumerate(train_loader):

        dataset = data[0]
        targets = data[1].float()
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #
        dataset = dataset.to(device)
        # Make predictions for this batch
        outputs = DeepFMModel(dataset)
        #print(outputs.shape)
        #output = torch.squeeze(outputs, 1)
        #print(output.shape)
        # Compute the loss and its gradients
        loss = loss_fn(outputs,targets.squeeze(1).to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        epoch_loss.append(loss.item())
    if(epoch % 1 == 0):
        print("The model output: ",outputs)
        print("The Target", targets)

        print(' Train epoch {} loss: {}'.format(epoch, np.mean(epoch_loss)))
        
        epoch_loss.append(loss.item())
    epoch_loss_value = np.mean(epoch_loss)
    Loss_list.append(epoch_loss_value)
    DeepFMModel.eval()
    epoch_valid_loss = []
    for batch, valid_data in enumerate(valid_loader):

        dataset = valid_data[0]
        targets = valid_data[1].float()
        outputs = DeepFMModel(dataset)
        loss = loss_fn(outputs,targets.squeeze(1).to(device))
        epoch_valid_loss.append(loss.item())
    if(epoch % 1 == 0):
        print(outputs.shape)
        print(' Valid epoch {} loss: {}'.format(epoch, np.mean(epoch_valid_loss)))
    epoch_valid_loss_value = np.mean(epoch_valid_loss)
    Valid_Loss_list.append(epoch_valid_loss_value)
    if(epoch_valid_loss_value < Best_loss):
        best_model = DeepFMModel
        Best_loss = epoch_valid_loss_value
#torch.save(model.state_dict(), 'Models/Baseline_MulitDim_model.pth')
#PATH = 'Models/DeepFM_model.pth'
#torch.save(best_model, PATH)

print("finished training")
print("Loss list = ", Loss_list)
