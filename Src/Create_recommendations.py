import torch
from Src.deepFM import CreateDataset, DatasetIter
import math

def Get_Recommendations(customer_id, model, test_set, test_full_set, test_full_data_path,chunksize,batch_size, num_recommendations = 6, iter_data = False):
    if(iter_data == False):
        #true_values = test_set[test_set['customer_id']==customer_id].article_id.unique()
        true_values = test_set.article_id.unique()
        #test_full_set = test_full_set[test_full_set['customer_id'] == customer_id]
        test_tensor = torch.tensor(test_full_set.fillna(0).to_numpy(), dtype = torch.int)
        #test_tensor = torch.tensor(test_full_set.values)
        test_dataset = CreateDataset(test_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = False)
        #last_batch = math.floor(len(test_full_set)/batch_size)
        for batch, (X,y) in enumerate(test_loader):

            if batch == 0:
                outputs, _ = model(X)
            #elif batch == (last_batch):
                #for i in range(X.shape[0]):
                    #outputs_temp, _ = model(X[i,:].unsqueeze(0))
                    #outputs = torch.cat((outputs,outputs_temp),0)
            else:
                outputs_temp, _ = model(X)
                outputs = torch.cat((outputs,outputs_temp),0)

        prob, idx = torch.topk(outputs, num_recommendations)
        items = test_tensor[idx,1]
        return items, true_values.astype(int)
    if(iter_data == True):
        true_values = test_set[test_set['customer_id']==customer_id].article_id.unique()
        #test_full_set = test_full_set[test_full_set['customer_id'] == customer_id]
        #print(test_tensor)
        test_dataset = DatasetIter(test_full_data_path, chunksize)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = False)
        last_batch = math.floor(len(test_dataset)/batch_size)
        for batch, (X,y) in enumerate(test_loader):
            print(batch)
            print(last_batch)
            if batch == 0:
                outputs = model(X)
            elif batch == (last_batch):
                for i in range(X.shape[0]):
                    outputs_temp = model(X[i,:].unsqueeze(0))
                    outputs = torch.cat((outputs,outputs_temp),0)
            else:
                outputs_temp = model(X)
                outputs = torch.cat((outputs,outputs_temp),0)
        print(outputs.shape)
        print(num_recommendations)
        print(customer_id)
        prob, idx = torch.topk(outputs, num_recommendations)
        items = test_tensor[idx,1]
        return items, true_values.astype(int)