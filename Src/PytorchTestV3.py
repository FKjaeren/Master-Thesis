import torch
#import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
import copy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#dtype = torch.float
#device = 
device = torch.device("mps")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("mps")
class CreateDataset(Dataset):
    def __init__(self, dataset, features, idx_variable):

        self.id = idx_variable
        self.features = features
        self.all_data = dataset
        self.device = device

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, row):
        #print("Før feature og ind til device ")

        features = torch.tensor(self.all_data[self.features].to_numpy(), dtype = torch.int)#, device=self.device)
        idx_variable = torch.tensor(self.all_data[self.id].to_numpy(), dtype = torch.int)#, device = self.device)
        #print("Hej ", idx_variable.device)
        all_data = torch.cat((idx_variable, features), dim = 1)
        #all_data = all_data.to(self.device)
        #print("device of dataset is: ",all_data.device)
        return all_data[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

class RecSysModel(torch.nn.Module):
    def __init__(self, Customer_data, Products_data, embedding_dim, batch_size, n_products, n_customers, n_prices, n_colours, n_departments,n_ages=111):
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
        self.customer_embedding = nn.Embedding(self.num_customers+2, embedding_dim)#, device = device)        
        self.product_embedding = nn.Embedding(self.num_products+2, embedding_dim)#, device = device)
        self.price_embedding = nn.Embedding(self.num_prices+2, embedding_dim)#, device = device)
        self.age_embedding = nn.Embedding(self.num_ages+2,embedding_dim)#, device = device)
        self.colour_embedding = nn.Embedding(self.num_colours+2, embedding_dim)#, device = device)
        self.department_embedding = nn.Embedding(self.num_departments+2, embedding_dim)#, device = device)


        self.All_Products = Products_data

        #self.out = nn.Linear(64,n_products+1)

    def monitor_metrics(self, output, target):
        output = output.detatch().numpy()
        target = target.detatch().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, Customer_data, Product_data):
        device = self.device
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


        output = torch.matmul((customer_embedding_final), torch.t(product_embedding_final))
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
        return recommendations, indexes


train_df = pd.read_csv('Data/Preprocessed/TrainData_enriched.csv')
#train_df = train_df.iloc[0:100000]
valid_df = pd.read_csv('Data/Preprocessed/ValidData_enriched.csv')
test_df = pd.read_csv('Data/Preprocessed/TestData_enriched.csv')

Customer_data = pd.read_csv('Data/Preprocessed/Customers_enriched.csv')
Product_data = pd.read_csv('Data/Preprocessed/Products_enriched.csv')

train_df = train_df.iloc[0:100000]

all_customers = pd.read_csv('Data/Raw/customers.csv')
all_products = pd.read_csv('Data/Raw/articles.csv')

num_customers = all_customers['customer_id'].nunique()
num_products = all_products['article_id'].nunique()
num_departments = all_products['department_name'].nunique()
num_colours = all_products['colour_group_name'].nunique()
max_age = 110


Customer_id_Encoder = preprocessing.LabelEncoder().fit(all_customers['customer_id'])
Product_Encoder = preprocessing.LabelEncoder().fit(all_products['article_id'])
Customer_data['customer_id'] = Customer_id_Encoder.transform(Customer_data['customer_id'])
#Product_data['article_id'] = Product_Encoder.transform(Product_data['article_id'])
#Product_data = Product_data.drop(columns=['prod_name'], axis = 1)


Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(all_products[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(all_products[['department_name']].to_numpy().reshape(-1, 1))
Age_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=110).fit(all_customers[['age']].to_numpy().reshape(-1, 1))

all_prices = np.concatenate((Product_data['price'].unique(),Customer_data['price'].unique()),axis = 0)
all_prices = np.unique(all_prices)
num_prices = len(all_prices)

Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(all_prices.reshape(-1, 1))


train_dataset = copy.deepcopy(train_df)
train_dataset['price'] = train_dataset['price'].round(decimals=4)
train_dataset.customer_id = Customer_id_Encoder.transform(train_df.customer_id.values)
train_dataset['product_id'] = Product_Encoder.transform(train_df.article_id.values)
train_dataset['colour_group_name'] = Colour_Encoder.transform(train_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
train_dataset['department_name'] = Department_encoder.transform(train_dataset[['department_name']].to_numpy().reshape(-1, 1))
train_dataset['age'] = Age_encoder.transform(train_dataset[['age']].to_numpy().reshape(-1,1))
train_dataset['price'] = Price_encoder.transform(train_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)



valid_df = valid_df[0:100000]

valid_dataset = copy.deepcopy(valid_df)
valid_dataset.customer_id = Customer_id_Encoder.transform(valid_df.customer_id.values)
valid_dataset['product_id'] = Product_Encoder.transform(valid_df.article_id.values)
valid_dataset['colour_group_name'] = Colour_Encoder.transform(valid_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
valid_dataset['department_name'] = Department_encoder.transform(valid_dataset[['department_name']].to_numpy().reshape(-1, 1))
valid_dataset['age'] = Age_encoder.transform(valid_dataset[['age']].to_numpy().reshape(-1,1))
valid_dataset['price'] = Price_encoder.transform(valid_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)


test_dataset = copy.deepcopy(test_df)
test_dataset.customer_id = Customer_id_Encoder.transform(test_df.customer_id.values)
test_dataset['product_id'] = Product_Encoder.transform(test_df.article_id.values)
test_dataset['colour_group_name'] = Colour_Encoder.transform(test_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
test_dataset['department_name'] = Department_encoder.transform(test_dataset[['department_name']].to_numpy().reshape(-1, 1))
test_dataset['age'] = Age_encoder.transform(test_dataset[['age']].to_numpy().reshape(-1,1))
test_dataset['price'] = Price_encoder.transform(test_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)

#test_dataset.to_csv('Data/Preprocessed/test_final.csv',index=False)

#Product_data['colour_group_name'] = Colour_Encoder.transform(Product_data[['colour_group_name']].to_numpy().reshape(-1, 1))
#Product_data['department_name'] = Department_encoder.transform(Product_data[['department_name']].to_numpy().reshape(-1, 1))
#Product_data = Product_data[['article_id','age','price','sales_channel_id','colour_group_name','department_name']]
Product_data['age'] = Product_data['age'].round(decimals = 0)
Product_data['age'] = Age_encoder.transform(Product_data[['age']].to_numpy().reshape(-1,1))
Product_data['price'] = Product_data['price'].round(decimals=4)
Product_data['price'] = Price_encoder.transform(Product_data[['price']].to_numpy().reshape(-1,1))

train_dataset = train_dataset[['customer_id','product_id','age','price','colour_group_name','department_name']]

Customer_data['colour_group_name'] = Customer_data['max_colour'].str.replace('colour_group_name_','')
Customer_data['department_name'] = Customer_data['max_department'].str.replace('department_name_','')
Customer_data = Customer_data.drop(columns=['max_colour','max_department'], axis = 1)

Customer_data['colour_group_name'] = Colour_Encoder.transform(Customer_data[['colour_group_name']].to_numpy().reshape(-1, 1))
Customer_data['department_name'] = Department_encoder.transform(Customer_data[['department_name']].to_numpy().reshape(-1, 1))
Customer_data['price'] = Customer_data['price'].round(decimals=4)
Customer_data['price'] = Price_encoder.transform(Customer_data[['price']].to_numpy().reshape(-1,1))

customer_dataset = CreateDataset(Customer_data,features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_dataset = CreateDataset(Product_data, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])


product_train_dataset = CreateDataset(train_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])
customer_train_dataset = CreateDataset(train_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_valid_dataset = CreateDataset(valid_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])
customer_valid_dataset = CreateDataset(valid_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

batch_size = 1024
embedding_dim = 64
model = RecSysModel(customer_dataset, product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_products=num_products+1,
                    n_customers=num_customers+1, n_prices=num_prices +1, n_colours=num_colours+1, n_departments=num_departments+1)
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
    for i, product_data_batch,customer_data_batch in zip(np.arange(1,product_train_dataset.shape()[0]),product_train_loader,customer_train_loader):
        #product_id = product_id.view(batch_size,1)
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
        loss = loss_fn(output,product_id)
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
    for batch, product_data_batch_valid, customer_data_batch_valid in zip(np.arange(1,product_valid_dataset.shape()[0]), product_valid_loader, customer_valid_loader):
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
PATH = 'Models/Baseline_MulitDim_model.pth'
torch.save(best_model, PATH)

#prof.stop()
print("finished training")
print("Loss list = ", Loss_list)

plt.plot(np.arange(1,len(Loss_list)+1), Loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training graph')
plt.show()
