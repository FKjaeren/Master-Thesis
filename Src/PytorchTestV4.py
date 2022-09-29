import torch
import platform
#import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
import copy
from torch.utils.data import Dataset, TensorDataset
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OS = platform.system()
if(OS == 'Darwin'):
    device = torch.device("cpu")
    #device = torch.device("mps")
elif(OS == "Windows"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif(OS == "Linux"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    print('The operating system is not reconized, therefore we could not set device type :(')

class CreateDataset(Dataset):
    def __init__(self, dataset):#, features, idx_variable):

        #self.id = idx_variable
        #self.features = features
        self.all_data = dataset
        #self.device = device

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, row):
        #print("Før feature og ind til device ")

        #features = torch.tensor(self.all_data[self.features].to_numpy(), dtype = torch.int)#, device=self.device)
        #idx_variable = torch.tensor(self.all_data[self.id].to_numpy(), dtype = torch.int)#, device = self.device)
        #print("Hej ", idx_variable.device)
        #all_data = torch.cat((idx_variable, features), dim = 1)
        #all_data = all_data.to(self.device)
        #print("device of dataset is: ",all_data.device)
        return self.all_data[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

class RecSysModel(torch.nn.Module):
    def __init__(self, Customer_data, Products_data, embedding_dim, batch_size, n_products, n_customers, n_prices, n_colours, n_departments,n_ages=111,device=device):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_customers = n_customers
        self.num_products = n_products
        self.num_prices = n_prices
        self.num_ages = n_ages
        self.num_colours = n_colours
        self.num_departments = n_departments
        self.customer_embedding = nn.Embedding(self.num_customers+2, embedding_dim).to(device)    
        self.product_embedding = nn.Embedding(self.num_products+2, embedding_dim).to(device)
        self.price_embedding = nn.Embedding(self.num_prices+2, embedding_dim).to(device)
        self.age_embedding = nn.Embedding(self.num_ages+2,embedding_dim).to(device)
        self.colour_embedding = nn.Embedding(self.num_colours+2, embedding_dim).to(device)
        self.department_embedding = nn.Embedding(self.num_departments+2, embedding_dim).to(device)


        self.All_Products = Products_data#.to(device)

        #self.out = nn.Linear(64,n_products+1)

    def monitor_metrics(self, output, target):
        output = output.detatch().numpy()
        target = target.detatch().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, Customer_data, Product_data):
        device = self.device
        All_products = self.All_Products[:,:].to(device)
        customer_embedding = self.customer_embedding(Customer_data[:,0])
        price_embedding = self.price_embedding(Customer_data[:,1])
        age_embedding = self.age_embedding(Customer_data[:,2])
        colour_embedding = self.colour_embedding(Customer_data[:,3])
        department_embedding = self.department_embedding(Customer_data[:,4])
        customer_embedding_final = torch.cat((customer_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1).to(device)
        product_embedding = self.product_embedding(All_products[:,0])
        price_embedding = self.price_embedding(All_products[:,1])
        age_embedding = self.age_embedding(All_products[:,2])
        colour_embedding = self.colour_embedding(All_products[:,3])
        department_embedding = self.department_embedding(All_products[:,4])
        product_embedding_final = torch.cat((product_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1).to(device)
        output = torch.matmul((customer_embedding_final), torch.t(product_embedding_final)).to(device)
        #calc_metrics = self.monitor_metrics(output,Product_data[:,0].view(1,-1))
        return output#, calc_metrics

    def CustomerItemRecommendation(self, Customer_data, k):
        customer_embedding = self.customer_embedding(Customer_data[:,0])
        price_embedding = self.price_embedding(Customer_data[:,1])
        age_embedding = self.age_embedding(Customer_data[:,2])
        colour_embedding = self.colour_embedding(Customer_data[:,3])
        department_embedding = self.department_embedding(Customer_data[:,4])
        customer_embedding_final = torch.cat((customer_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1)

        product_embedding = self.product_embedding(self.All_Products[:,0])
        price_embedding = self.price_embedding(self.All_Products[:,1])
        age_embedding = self.age_embedding(self.All_Products[:,2])
        colour_embedding = self.colour_embedding(self.All_Products[:,3])
        department_embedding = self.department_embedding(self.All_Products[:,4])
        product_embedding_final = torch.cat((product_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1)

        matrixfactorization = torch.matmul((customer_embedding_final), torch.t(product_embedding_final))
        recommendations, indexes = torch.topk(matrixfactorization, k = k)
        return recommendations, indexes, matrixfactorization


train_df = pd.read_csv('Data/Preprocessed/TrainData_enriched_subset.csv')
#train_df = train_df.iloc[0:100000]
valid_df = pd.read_csv('Data/Preprocessed/ValidData_enriched_subset.csv')
#test_df = pd.read_csv('Data/Preprocessed/TestData_enriched.csv')
#valid_df = valid_df.iloc[0:50000]

Customer_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame.csv')
Product_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame.csv')
transactions_df = pd.read_csv('Data/Preprocessed/transactions_df_numeric.csv')

splitrange = round(0.75*len(transactions_df['customer_id']))
splitrange2 = round(0.95*len(transactions_df['customer_id']))
train = transactions_df.iloc[:splitrange]
valid = transactions_df.iloc[splitrange+1:splitrange2]
test = transactions_df.iloc[splitrange2:]
Customer_data_tensor = torch.tensor(Customer_data[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
Product_data_tensor = torch.tensor(Product_data[['article_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
customer_train_tensor = torch.tensor(train_dataset[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
product_train_tensor = torch.tensor(train_dataset[['article_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)

customer_valid_tensor = torch.tensor(valid_dataset[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
product_valid_tensor = torch.tensor(valid_dataset[['article_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)

"""
customer_dataset = TensorDataset(Customer_data_tensor)
product_dataset = TensorDataset(Product_data_tensor)
product_train_dataset = TensorDataset(product_train_tensor)
customer_train_dataset = TensorDataset(customer_train_tensor)
product_valid_dataset = TensorDataset(customer_valid_tensor)
customer_valid_dataset = TensorDataset(customer_valid_tensor)

"""
customer_dataset = CreateDataset(Customer_data_tensor)#,features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_dataset = CreateDataset(Product_data_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])


product_train_dataset = CreateDataset(product_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
customer_train_dataset = CreateDataset(customer_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_valid_dataset = CreateDataset(customer_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
customer_valid_dataset = CreateDataset(customer_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])


batch_size = 1024
embedding_dim = 64
model = RecSysModel(customer_dataset, product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_products=num_products+1,
                    n_customers=num_customers+1, n_prices=num_prices +1, n_colours=num_colours+1, n_departments=num_departments+1,device=device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001, lr = 0.005)
model =model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 1


#Training in batches:
product_train_loader = torch.utils.data.DataLoader(product_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)
customer_train_loader = torch.utils.data.DataLoader(customer_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

product_valid_loader = torch.utils.data.DataLoader(product_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
customer_valid_loader = torch.utils.data.DataLoader(customer_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

#Num_classes = len(Product_data['product_id'])
dataiter = iter(product_train_loader)
dataset = next(dataiter)

#prof = torch.profiler.profile(
#        schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=2),
#        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/BaselineModel'),
#        record_shapes=True,
#        with_stack=True)
#prof.start()
Loss_list = []
Valid_Loss_list = []
Best_loss = np.infty
for epoch in range(1,num_epochs+1):
    running_loss = 0.
    epoch_loss = []
    model.train()
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # Every data instance is an input + label pair
    for i, product_data_batch,customer_data_batch in zip(np.arange(1,product_train_tensor.shape[0]),product_train_loader,customer_train_loader):
        #product_id = product_id.view(batch_size,1)
        #print(product_data_batch)
        product_id = product_data_batch[:,0]
        product_id = product_id.type(torch.long)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        #
        customer_data_batch = customer_data_batch.to(device)
        product_data_batch = product_data_batch.to(device)
        # Make predictions for this batch
        outputs = model(customer_data_batch, product_data_batch)
        output = torch.squeeze(outputs, 1)
        # Compute the loss and its gradients
        loss = loss_fn(output,product_id.to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        epoch_loss.append(loss.item())
        if(i % 10 == 0):
            print(' Train batch {} loss: {}'.format(i, np.mean(epoch_loss)))
        
        epoch_loss.append(loss.item())
    epoch_loss_value = np.mean(epoch_loss)
    Loss_list.append(epoch_loss_value)
    model.eval()
    epoch_valid_loss = []
    for batch, product_data_batch_valid, customer_data_batch_valid in zip(np.arange(1,product_train_tensor.shape[0]), product_valid_loader, customer_valid_loader):
        product_id = product_data_batch_valid[:,0].type(torch.long)
        outputs = model(customer_data_batch_valid, product_data_batch_valid)
        output = torch.squeeze(outputs, 1)
        loss = loss_fn(output,product_id)
        epoch_valid_loss.append(loss.item())
        if(batch % 10 == 0):
            print(' Valid batch {} loss: {}'.format(batch, np.mean(epoch_valid_loss)))
    epoch_valid_loss_value = np.mean(epoch_valid_loss)
    Valid_Loss_list.append(epoch_valid_loss_value)
    if(epoch_valid_loss_value < Best_loss):
        best_model = model
        Best_loss = epoch_valid_loss_value
#torch.save(model.state_dict(), 'Models/Baseline_MulitDim_model.pth')
#PATH = 'Models/Baseline_MulitDim_model.pth'
#torch.save(best_model, PATH)

#prof.stop()

"""
print("finished training")
print("Loss list = ", Loss_list)

plt.plot(np.arange(1,len(Loss_list)+1), Loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training graph')
plt.show()

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
   with record_function("model_inference"):
      model(customer_data_batch_valid, product_data_batch_valid)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
"""