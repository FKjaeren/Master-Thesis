import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
import copy
import os
from Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron#, FeaturesLinear

class DatasetIter(Dataset):
    def __init__(self, csv_path, chunkSize):
        self.chunksize = chunkSize
        self.reader = pd.read_csv(csv_path, sep=',', chunksize=self.chunksize, header=None, iterator=True)

    def __len__(self):
        return self.chunksize

    def __getitem__(self, index):
        data = self.reader.get_chunk(self.chunksize)
        tensorData = torch.tensor(data.values, dtype=torch.int)
        inputs = tensorData[:,:-1]
        labels = tensorData[:,-1]
        return inputs, labels

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

class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A Pytorch implementation of DeepFM.
    Reference:
    H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, n_unique_dict, device, batch_size, dropout):
        super().__init__()
        #mlp_dims = [22,107,47]
        mlp_dims = [16,32,16]
        #self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(embedding_dim = embed_dim,num_fields=field_dims ,batch_size= batch_size, n_unique_dict=n_unique_dict, device = device)
        self.embed_output_dim = (len(field_dims)-1) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout=dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = (self.fm(embed_x)*1.2737) + (self.mlp(embed_x.view(-1, self.embed_output_dim))*1.341)
    def Reccomend_topk(x, k):
        item_idx = torch.topk(x,k)
        return item_idx
