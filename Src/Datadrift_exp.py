import torchdrift
from Src.deepFM import DeepFactorizationMachineModel
import torch
import pickle
from torch.utils.data import Dataset
from torch import nn
import copy
import pandas as pd

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


inputs, _ = next(iter(valid_loader))
inputs_ood = corruption_function(inputs)

feature_extractor = copy.deepcopy(model)

feature_extractor.embed_x = torch.nn.Identity()

drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
torchdrift.utils.fit(valid_loader, feature_extractor, drift_detector, num_batches=1)
drift_detection_model = torch.nn.Sequential(
    feature_extractor,
    drift_detector
)