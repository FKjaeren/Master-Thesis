import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
import copy
import logging
import time
import yaml
from yaml.loader import SafeLoader
from FMModel import FactorizationMachineModel

# Open the file and load the file
with open('config/experiment/train_FM.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)
def main():
    #Create dataset class
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

    #Load data
    train_df = pd.read_csv('Data/Preprocessed/train_df_subset.csv')
    valid_df = pd.read_csv('Data/Preprocessed/valid_df_subset.csv')
    test_df = pd.read_csv('Data/Preprocessed/test_df_subset.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Convert data to torch tensor
    train_tensor = torch.tensor(train_df.fillna(0).to_numpy(), dtype = torch.int)    
    valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)
    test_tensor = torch.tensor(test_df.fillna(0).to_numpy(), dtype = torch.int)
    #Convert data to dataset class
    train_dataset = CreateDataset(train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    valid_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    test_dataset = CreateDataset(test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

    batch_size = hparams["batch_size"]

    #dataset_shapes = {'train_shape':train_tensor.shape,'valid_shape':valid_tensor.shape,'test_shape':test_tensor.shape}

    #Load data in batches
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

    with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)
    #Define model
    FMModel = FactorizationMachineModel(field_dims = train_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = device)
    optimizer = torch.optim.Adam(FMModel.parameters(), weight_decay=hparams["weight_decay"], lr = hparams["lr"])
    if hparams["pos_weight"] == "data_scale":
        pos_weight = train_df.target.value_counts()[0] / train_df.target.value_counts()[1]
    else:
        pos_weight = hparams["pow_weight"]

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
    loss_fn_val = torch.nn.BCEWithLogitsLoss()
    #Initialize weights
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


    FMModel.apply(init_weights)

    num_epochs = hparams["num_epochs"]
    res = []
    Loss_list = []
    Valid_Loss_list = []
    Val_acc_list = []
    Train_Acc_list = []
    Best_loss = np.infty
    #Train
    for epoch in range(1,num_epochs+1):
        start = time.time()

        #print(epoch)
        running_loss = 0.
        running_loss_val = 0.
        epoch_loss = []
        epoch_acc = []
        FMModel.train()

        for batch, (X,y) in enumerate(train_loader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            #
            dataset = X.to(device)
            # Make predictions for this batch
            outputs, loss_output = FMModel(dataset)
            #if(torch.isnan(X).sum() > 0):
            #    print("SE her Values with nan in X: ",X[torch.isnan(X)])
            #if(torch.isnan(outputs).sum() > 0):
            #    print("Values with nan in outputs: ",outputs[torch.isnan(outputs)])
            #    print("And the batch is: ", batch)
            # Compute the loss and its gradients

            loss = loss_fn(loss_output,y.squeeze().to(device))
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            running_loss += loss 
                # Gather data and report
            epoch_loss.append(loss.item())
            predictions = outputs.detach().apply_(lambda x: 1 if x > 0.5 else 0)
            Train_acc = (1-abs(torch.sum(y.squeeze() - predictions).item())/len(y))*100
            Train_Acc_list.append(Train_acc)
            #if(batch % 500 == 0):
            #    print(' Train batch {} loss: {}'.format(batch, np.mean(epoch_loss)))
        if(epoch % 1 == 0):
            print(' Train epoch {} loss: {}'.format(epoch, np.mean(epoch_loss)))
        #Find best performing model
        epoch_loss_value = np.mean(epoch_loss)
        Loss_list.append(epoch_loss_value)
        FMModel.eval()
        epoch_valid_loss = []
        for batch, (X_valid,y_valid) in enumerate(valid_loader):
            outputs, loss_output = FMModel(X_valid)
            loss_val = loss_fn_val(loss_output,y_valid.squeeze())
            epoch_valid_loss.append(loss_val.item())
            running_loss_val += loss_val
            predictions = outputs.detach().apply_(lambda x: 1 if x > 0.5 else 0)
            Val_acc = (1-abs(torch.sum(y_valid.squeeze() - predictions).item())/len(y_valid))*100
            Val_acc_list.append(Val_acc)

        if(epoch % 1 == 0):
            print(' Valid epoch {} loss: {}'.format(epoch, np.mean(epoch_valid_loss)))

        epoch_valid_loss_value = np.mean(epoch_valid_loss)
        Valid_Loss_list.append(epoch_valid_loss_value)
        if(epoch_valid_loss_value < Best_loss):
            best_model = copy.deepcopy(FMModel)
            Best_loss = epoch_valid_loss_value

        end = time.time()
        res.append(end - start)
    res = np.array(res)
    PATH = hparams["model_path"]
    torch.save(best_model.state_dict(), PATH)

    print("finished training")
    print("Loss list = ", Loss_list)
    print("Training accuracy is: ", (sum(Train_Acc_list)/len(Train_Acc_list)))
    print("Validation accuracy is: ", (sum(Val_acc_list)/len(Val_acc_list)))
if __name__ == '__main__':
    main()
