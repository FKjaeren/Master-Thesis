import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import pickle
import copy
from Src.Layers import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, LinearLayer
import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

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
        #tensorData = torch.tensor(Dataset.values, dtype=torch.int)
        self.dataset = dataset[:,:-1]
        self.targets = dataset[:,-1:]#.float()

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

    def __init__(self, field_dims, hparams, n_unique_dict, device):
        super().__init__()
        mlp_dims = [hparams["latent_dim1"],hparams["latent_dim2"],hparams["latent_dim3"]]
        self.linear = LinearLayer()
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(embedding_dim = hparams["embed_dim"],num_fields=field_dims, n_unique_dict=n_unique_dict, device = device)
        self.embed_output_dim = (len(field_dims)-1) * hparams["embed_dim"]
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout=hparams["dropout"])

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)

        x = ((self.linear(embed_x.view(-1, self.embed_output_dim)).unsqueeze(dim=1)+self.fm(embed_x))*hparams["fm_weight"]) + (self.mlp(embed_x.view(-1, self.embed_output_dim))*hparams["mlp_weight"])

        return torch.sigmoid(x.squeeze(1)), x.squeeze(1)
    def Reccomend_topk(x, k):
        item_idx = torch.topk(x,k)
        return item_idx