import torchdrift
from Src.deepFM import DeepFactorizationMachineModel
import torch
import pickle
from torch.utils.data import Dataset
from torch import nn
import copy
import pandas as pd
import pytorch_lightning as pl

valid_df = pd.read_csv('Data/Preprocessed/valid_df_subset.csv')

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)



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



batch_size = 128
valid_tensor = torch.tensor(valid_df.fillna(0).to_numpy(), dtype = torch.int)

valid_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)


embedding_dim = 26
dropout=0.2677
device = 'cpu'
path = 'Models/DeepFM_model_Final.pth'
model = DeepFactorizationMachineModel(field_dims = valid_df.columns, embed_dim=embedding_dim, n_unique_dict = number_uniques_dict, device = device, batch_size=batch_size,dropout=dropout)

model.load_state_dict(torch.load(path))



def corruption_function(x: torch.Tensor):
    return torchdrift.data.functional.gaussian_blur(x, severity=2)



inputs, _  = next(iter(valid_loader))

def collate_fn(batch):
    batch = torch.utils.data._utils.collate.default_collate(batch)
    batch = ((batch[0]), *batch[1:])
    return batch
col = collate_fn(inputs)
inputs_ood = corruption_function(col)




feature_extractor = copy.deepcopy(model)

#feature_extractor.embedding.fc = torch.nn.Identity()




class OurDataModule(pl.LightningDataModule):
    def __init__(self, parent: Optional['OurDataModule']=None, additional_transform=None):
        if parent is None:
            
            self.val_dataset = CreateDataset(valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

            
            self.val_batch_size = 128
            self.additional_transform = None
        else:
            self.val_dataset = parent.val_dataset
            self.val_batch_size = parent.val_batch_size
            self.additional_transform = additional_transform
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
                                           num_workers=4, shuffle=False, collate_fn=self.collate_fn)

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


datamodule = OurDataModule()











detector = torchdrift.detectors.ks.KSDriftDetector()



odd_input = (inputs + torch.randn(inputs.shape[0], inputs.shape[1])).type(torch.int32)

odd_dataset = CreateDataset(odd_input)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])


odd_loader = torch.utils.data.DataLoader(odd_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)


#odd_input = torchdrift.data.functional.gaussian_noise(inputs, severity= ) 

ood_ratio = 0.8
sample_size = 1
experiment = torchdrift.utils.DriftDetectionExperiment(detector, feature_extractor.embedding, ood_ratio=ood_ratio, sample_size=sample_size)
experiment.post_training(valid_loader)
auc, (fp, tp) = experiment.evaluate(valid_loader, odd_loader)
pyplot.plot(fp, tp)
pyplot.title(label=f'{detector}, {red}\n$p_{{ood}}$={ood_ratio:.2f}, N={sample_size} AUC={auc:.3f}')
pyplot.show()