import numpy as np
import pandas as pd
import torch
from Src.deepFM import CreateDataset

PATH = 'Models/DeepFM_model.pth'
model = torch.load(PATH)
device = torch.device('cpu')


test_dataset = pd.read_csv('Data/Preprocessed/test_df.csv')
articles_df = pd.read_csv('Data/Raw/articles.csv')
num_products = articles_df.article_id.nunique()

test_tensor = torch.tensor(test_dataset.fillna(0).to_numpy(), dtype = torch.int)
test_dataset = CreateDataset(test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
batch_size = num_products
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)

accuracy = []

test_group = test_dataset[['customer_id','article_id']].groupby(['customer_id']).nunique()
test_group2 = test_dataset[['customer_id','article_id']].groupby(['customer_id'])['article_id'].apply(list)

model.eval()
for batch, (X,y) in enumerate(test_loader):
    dataset = X.to(device)
    outputs = model(dataset)
    topk_recommendations = model.Reccomend_topk(outputs,6)
    customer_id = X[0,0]
    true_values = torch.tensor(test_group2.loc[customer_id], dtype = torch.int)
    check = any(item in topk_recommendations for item in true_values)
    if(check == True):
        accuracy.append(1)
    else:
        accuracy.append(0)

print("Accuracy on the test dataet is: ", np.mean(accuracy)*100, "%")

