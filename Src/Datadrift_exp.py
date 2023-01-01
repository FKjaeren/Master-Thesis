import torchdrift
from deepFM import DeepFactorizationMachineModel
import torch
import pickle
from torch.utils.data import Dataset
from torch import nn
import copy
import pandas as pd
import pytorch_lightning as pl

import yaml
from yaml.loader import SafeLoader

# Get pytorch dataset
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

# load data
valid_df = pd.read_csv('Data/Preprocessed/valid_df_subset.csv')[:20000]
valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)


valid_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)


# Datadrift default dataloader 
class OurDataModule(pl.LightningDataModule):
    def __init__(self, dataset, parent=None, additional_transform=None):
        if parent is None:
            print("hej1")
            self.val_dataset = dataset

            print("hej)2")
            self.val_batch_size = 128
            self.additional_transform = None
        else:
            print("Hej 3")
            self.val_dataset = parent.val_dataset
            self.val_batch_size = parent.val_batch_size
            self.additional_transform = additional_transform
            print("Hej4 ")
        if additional_transform is not None:
            self.additional_transform = additional_transform

        self.prepare_data()
        self.setup('fit')
        self.setup('test')

    def setup(self, typ):
        pass

    def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        if self.additional_transform:
            batch = (self.additional_transform(batch[0]), *batch[1:])
        return batch
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                                           num_workers=0, shuffle=False, collate_fn=self.collate_fn)

    def default_dataloader(self, batch_size=None, num_samples=None, shuffle=True):
        dataset = self.val_dataset
        if batch_size is None:
            batch_size = self.val_batch_size
        replacement = num_samples is not None
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=replacement, num_samples=num_samples)
        else:
            sampler = None
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                           collate_fn=self.collate_fn)


#################################################
# new corruption function

datamodule = OurDataModule(valid_dataset)
# Corrupt input data
def corruption_function(x: torch.Tensor):
    print(x.shape)
    return x + abs((torch.randint(low=0,high=2, size =(x.shape[0], x.shape[1])).type(torch.int32)))


with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)


with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)


# load model
device = 'cpu'
path = 'Models/DeepFM_modelV2.pth'

DeepFMModel = torch.load(path)

# detector and feature extractor 

feature_extractor = copy.deepcopy(DeepFMModel)

# only take the relevant information from the model embeddings
class model(nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.basemodel = basemodel
    def forward(self, x):
        # batch size and everything else in dim 2
        return self.basemodel(x).reshape(x.shape[0], -1)

newmodel = model(feature_extractor.embedding)

drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

# Make expperiment 
od_model = drift_detector
ind_datamodule = datamodule
ood_datamodule = OurDataModule(valid_dataset, parent=datamodule, additional_transform=corruption_function)

ood_ratio = 0.8
sample_size = 5
experiment = torchdrift.utils.DriftDetectionExperiment(od_model, newmodel, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(datamodule.val_dataloader())
auc, (fp, tp) = experiment.evaluate(ind_datamodule, ood_datamodule)



print(auc)
print(fp)
print(tp)



