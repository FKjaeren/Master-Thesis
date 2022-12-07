import torchdrift
from Src.deepFM import DeepFactorizationMachineModel
import torch
import pickle
from torch.utils.data import Dataset
from torch import nn
import copy
import pandas as pd
import pytorch_lightning as pl



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


valid_df = pd.read_csv('Data/Preprocessed/valid_df_subset.csv')[0:10000]
valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)


valid_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

#valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)



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


datamodule = OurDataModule(valid_dataset)
#torch.utils.data.DataLoader(valid_dataset, batch_size=128, num_workers=1, shuffle=False)


def corruption_function(x: torch.Tensor):
    print(x.shape)
    return abs((torch.randint(low=0,high=2, size =(x.shape[0], x.shape[1])).type(torch.int32)))


ind_datamodule = datamodule
ood_datamodule = OurDataModule(valid_dataset, parent=datamodule, additional_transform=corruption_function)

#odd_datamodule = (ind_datamodule + torch.randn(ind_datamodule.shape[0], ind_datamodule.shape[1])).type(torch.int32)




with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)


batch_size = 128
embedding_dim = 26
dropout=0.2677
device = 'cpu'
path = 'Models/DeepFM_model_Final.pth'
model = DeepFactorizationMachineModel(field_dims = valid_df.columns, embed_dim=embedding_dim, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size,dropout=dropout)

model.load_state_dict(torch.load(path))





detector = torchdrift.detectors.ks.KSDriftDetector()



""" odd_input = (inputs + torch.randn(inputs.shape[0], inputs.shape[1])).type(torch.int32)

odd_dataset = CreateDataset(odd_input)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])


odd_loader = torch.utils.data.DataLoader(odd_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
 """

#odd_input = torchdrift.data.functional.gaussian_noise(inputs, severity= ) 


feature_extractor = copy.deepcopy(model)

class model(nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.basemodel = basemodel
    def forward(self, x):
        # vi vil have batch size og alt andet i dim 2
        return self.basemodel(x).reshape(x.shape[0], -1)

newmodel = model(feature_extractor.embedding)


#torchdrift.utils.fit(datamodule.val_dataloader(), newmodel, detector)
drift_detector = torchdrift.detectors.KernelMMDDriftDetector()


od_model = drift_detector
ind_datamodule = datamodule
ood_datamodule = OurDataModule(valid_dataset, parent=datamodule, additional_transform=corruption_function)

ood_ratio = 0.8
sample_size = 10
experiment = torchdrift.utils.DriftDetectionExperiment(od_model, newmodel, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(datamodule.val_dataloader())
auc, (fp, tp) = experiment.evaluate(ind_datamodule, ood_datamodule)









#x, y = next(iter(datamodule.val_dataloader()))
ood_ratio = 0.8
sample_size = 2
experiment = torchdrift.utils.DriftDetectionExperiment(detector, newmodel, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(datamodule.val_dataloader())
auc, (fp, tp) = experiment.evaluate(ind_datamodule, ood_datamodule)
import matplotlib.pyplot as plt
plt.plot(fp, tp)

plt.show()



#################################################
# new corruption function

datamodule = OurDataModule(valid_dataset)
#torch.utils.data.DataLoader(valid_dataset, batch_size=128, num_workers=1, shuffle=False)


def corruption_function(x: torch.Tensor):
    print(x.shape)
    return x + abs((torch.randint(low=0,high=2, size =(x.shape[0], x.shape[1])).type(torch.int32)))



ind_datamodule = datamodule
ood_datamodule = OurDataModule(valid_dataset, parent=datamodule, additional_transform=corruption_function)

#odd_datamodule = (ind_datamodule + torch.randn(ind_datamodule.shape[0], ind_datamodule.shape[1])).type(torch.int32)




with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)


batch_size = 128
embedding_dim = 26
dropout=0.2677
device = 'cpu'
path = 'Models/DeepFM_model_Final.pth'
model = DeepFactorizationMachineModel(field_dims = valid_df.columns, embed_dim=embedding_dim, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size,dropout=dropout)

model.load_state_dict(torch.load(path))





detector = torchdrift.detectors.ks.KSDriftDetector()



""" odd_input = (inputs + torch.randn(inputs.shape[0], inputs.shape[1])).type(torch.int32)

odd_dataset = CreateDataset(odd_input)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])


odd_loader = torch.utils.data.DataLoader(odd_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
 """

#odd_input = torchdrift.data.functional.gaussian_noise(inputs, severity= ) 


feature_extractor = copy.deepcopy(model)

class model(nn.Module):
    def __init__(self, basemodel):
        super().__init__()
        self.basemodel = basemodel
    def forward(self, x):
        # vi vil have batch size og alt andet i dim 2
        return self.basemodel(x).reshape(x.shape[0], -1)

newmodel = model(feature_extractor.embedding)


#torchdrift.utils.fit(datamodule.val_dataloader(), newmodel, detector)
drift_detector = torchdrift.detectors.KernelMMDDriftDetector()


od_model = drift_detector
ind_datamodule = datamodule
ood_datamodule = OurDataModule(valid_dataset, parent=datamodule, additional_transform=corruption_function)

ood_ratio = 0.8
sample_size = 5
experiment = torchdrift.utils.DriftDetectionExperiment(od_model, newmodel, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(datamodule.val_dataloader())
auc, (fp, tp) = experiment.evaluate(ind_datamodule, ood_datamodule)









#x, y = next(iter(datamodule.val_dataloader()))
ood_ratio = 0.8
sample_size = 5
experiment = torchdrift.utils.DriftDetectionExperiment(detector, newmodel, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(datamodule.val_dataloader())
auc, (fp, tp) = experiment.evaluate(ind_datamodule, ood_datamodule)
import matplotlib.pyplot as plt
plt.plot(fp, tp)

plt.show()