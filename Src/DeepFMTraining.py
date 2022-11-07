import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
import copy
from Src.Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron#, FeaturesLinear
import hydra
#from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import time


log = logging.getLogger(__name__)
@hydra.main(config_path="config", config_name='config.yaml')
hparams = config.experiment
device = torch.device("cuda" if hparams['cuda'] else "cpu")
log.info(f'hparameters:  {hparams}')


wandb.init(project="MasterThesis", entity="frederikogjonesmaster")



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


train_df = pd.read_csv('Data/Preprocessed/train_df.csv')
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

    def __init__(self, field_dims, hparams, mlp_dims, n_unique_dict, device, batch_size):
        super().__init__()
        mlp_dims = [hparams["latent_dim1"], hparams["latent_dim2"], hparams["latent_dim3"]]
        #self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(embedding_dim = hparams["embed_dim"],num_fields=field_dims ,batch_size= batch_size, n_unique_dict=n_unique_dict, device = device)
        self.embed_output_dim = (len(field_dims)-1) * hparams["embed_dim"]
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, hparams["dropout"])

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



#embedding_dim = 16
DeepFMModel = DeepFactorizationMachineModel(field_dims = train_df.columns, hparams = hparams, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size)
optimizer = torch.optim.Adam(DeepFMModel.parameters(), weight_decay=hparams["weight_decay"], lr = hparams["lr"])
#pos_weight = train_df.target.value_counts()[0] / train_df.target.value_counts()[1]
#pos_weight = 100000
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(hparams["pos_weight"]))
#loss_fn = torch.nn.BCEWithLogitsLoss()
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Embedding):
        m.weight.data.normal_(mean=0.0, std=1.0)
        #m.weight.data.fill_(0.0)
        if m.padding_idx is not None:
            m.weight.data[m.padding_idx].zero_()


DeepFMModel.apply(init_weights)
wandb.watch(DeepFMModel, log_freq=500)

num_epochs = 2
res = []
Loss_list = []
Valid_Loss_list = []
Best_loss = np.infty
for epoch in range(1,num_epochs+1):
    start = time.time()

    print(epoch)
    running_loss = 0.
    running_loss_val = 0.
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
        running_loss += loss 
            # Gather data and report
        epoch_loss.append(loss.item())
        #if(batch % 500 == 0):
        #    print(' Train batch {} loss: {}'.format(batch, np.mean(epoch_loss)))
        wandb.log({"train_loss": loss})
    if(epoch % 1 == 0):
        print(' Train epoch {} loss: {}'.format(epoch, np.mean(epoch_loss)))
        log.info(f"at epoch: {epoch} the Training loss is : {running_loss/len(train_loader)}") 

        epoch_loss.append(loss.item())


    epoch_loss_value = np.mean(epoch_loss)
    Loss_list.append(epoch_loss_value)
    DeepFMModel.eval()
    epoch_valid_loss = []
    for batch, (X_valid,y_valid) in enumerate(valid_loader):
        outputs = DeepFMModel(X_valid)
        loss_val = loss_fn(outputs,y_valid.squeeze())
        epoch_valid_loss.append(loss_val.item())
        running_loss_val += loss_val 

        wandb.log({"val_loss": loss_val})

    if(epoch % 1 == 0):
        print(' Valid epoch {} loss: {}'.format(epoch, np.mean(epoch_valid_loss)))
        log.info(f"at epoch: {epoch} the Validation loss is : {running_loss_val/len(valid_loader)}") 

    epoch_valid_loss_value = np.mean(epoch_valid_loss)
    Valid_Loss_list.append(epoch_valid_loss_value)
    if(epoch_valid_loss_value < Best_loss):
        best_model = copy.deepcopy(DeepFMModel)
        Best_loss = epoch_valid_loss_value

    end = time.time()
    res.append(end - start)
res = np.array(res)
log.info(f'Timing: {np.mean(res)} +- {np.std(res)}')
#PATH = 'Models/DeepFM_model.pth'
#torch.save(best_model, PATH)

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

