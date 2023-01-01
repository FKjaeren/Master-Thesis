#from math import prod
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
import pickle
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
""" OS = platform.system()
if(OS == 'Darwin'):
    device = torch.device("cpu")
    #device = torch.device("mps")
elif(OS == "Windows"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
elif(OS == "Linux"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    print('The operating system is not reconized, therefore we could not set device type :(') """

class CreateDataset(Dataset):
    def __init__(self, dataset):#, features, idx_variable):

        self.all_data = dataset

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, row):

        return self.all_data[row]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

# Construct MF model multi class architecture
class RecSysModel(torch.nn.Module):
    def __init__(self, Products_data, embedding_dim, batch_size, n_unique_dict,device,n_ages=111):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_unique_dict = n_unique_dict
        self.n_ages = n_ages

        self.customer_embedding = nn.Embedding(self.n_unique_dict['n_customers']+1, embedding_dim).to(device)    
        self.product_embedding = nn.Embedding(self.n_unique_dict['n_products']+1, embedding_dim).to(device)
        self.price_embedding = nn.Embedding(self.n_unique_dict['n_prices']+2, embedding_dim).to(device)
        self.age_embedding = nn.Embedding(self.n_ages+2,embedding_dim).to(device)
        self.colour_embedding = nn.Embedding(self.n_unique_dict['n_colours']+1, embedding_dim).to(device)
        self.department_embedding = nn.Embedding(self.n_unique_dict['n_departments']+1, embedding_dim).to(device)
        self.prod_name_embedding = nn.Embedding(self.n_unique_dict['n_prod_names']+1, embedding_dim).to(device)  
        self.sales_channel_id_embedding = nn.Embedding(self.n_unique_dict['n_sales_channels']+1, embedding_dim).to(device)  
        self.season_embedding = nn.Embedding(self.n_unique_dict['n_seasons']+1, embedding_dim).to(device)  
        self.day_embedding = nn.Embedding(self.n_unique_dict['n_days']+1, embedding_dim).to(device)  
        self.month_embedding = nn.Embedding(self.n_unique_dict['n_months']+1, embedding_dim).to(device)  
        self.year_embedding = nn.Embedding(self.n_unique_dict['n_year']+1, embedding_dim).to(device)  
        self.club_member_status_embedding = nn.Embedding(self.n_unique_dict['n_club_member_status']+1, embedding_dim).to(device)  
        self.fashion_news_frequency_embedding = nn.Embedding(self.n_unique_dict['n_fashion_news_frequency']+1, embedding_dim).to(device)  
        self.postal_code_embedding = nn.Embedding(self.n_unique_dict['n_postal']+1, embedding_dim).to(device)  
        self.graphical_embedding = nn.Embedding(self.n_unique_dict['n_graphical']+1, embedding_dim).to(device)  

        self.All_Products = Products_data#.to(device)


    def monitor_metrics(self, output, target):
        output = output.detatch().numpy()
        target = target.detatch().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, Customer_data, Product_data):
        device = self.device
        All_products = self.All_Products[:,:].to(device)
        customer_embedding = self.customer_embedding(Customer_data[:,0])
        club_membership_embedding = self.club_member_status_embedding(Customer_data[:,1])
        fashion_news_embedding = self.fashion_news_frequency_embedding(Customer_data[:,2])
        age_embedding = self.age_embedding(Customer_data[:,3])
        postal_code_embedding = self.postal_code_embedding(Customer_data[:,4])
        price_embedding = self.price_embedding(Customer_data[:,5])
        sales_channel_embedding = self.sales_channel_id_embedding(Customer_data[:,6])
        season_embedding = self.season_embedding(Customer_data[:,7])
        day_embedding = self.day_embedding(Customer_data[:,8])
        month_embbeding = self.month_embedding(Customer_data[:,9])
        year_embedding = self.year_embedding(Customer_data[:,10])
        prod_name_embedding = self.prod_name_embedding(Customer_data[:,11])
        graphical_embedding = self.graphical_embedding(Customer_data[:,12])
        colour_embedding = self.colour_embedding(Customer_data[:,13])
        department_embedding = self.department_embedding(Customer_data[:,14])
        customer_embedding_final = torch.cat((customer_embedding, prod_name_embedding, graphical_embedding, colour_embedding, department_embedding,
                                            price_embedding, sales_channel_embedding, season_embedding, day_embedding, month_embbeding, year_embedding,
                                            age_embedding, club_membership_embedding, fashion_news_embedding, postal_code_embedding), dim = 1).to(device)
        product_embedding = self.product_embedding(All_products[:,0])
        club_membership_embedding = self.club_member_status_embedding(All_products[:,1])
        fashion_news_embedding = self.fashion_news_frequency_embedding(All_products[:,2])
        age_embedding = self.age_embedding(All_products[:,3])
        postal_code_embedding = self.postal_code_embedding(All_products[:,4])
        price_embedding = self.price_embedding(All_products[:,5])
        sales_channel_embedding = self.sales_channel_id_embedding(All_products[:,6])
        season_embedding = self.season_embedding(All_products[:,7])
        day_embedding = self.day_embedding(All_products[:,8])
        month_embbeding = self.month_embedding(All_products[:,9])
        year_embedding = self.year_embedding(All_products[:,10])
        prod_name_embedding = self.prod_name_embedding(All_products[:,11])
        graphical_embedding = self.graphical_embedding(All_products[:,12])
        colour_embedding = self.colour_embedding(All_products[:,13])
        department_embedding = self.department_embedding(All_products[:,14])
        #index_embedding = self.index_group_name_embedding(All_products[:,18])

        product_embedding_final = torch.cat((product_embedding, prod_name_embedding, graphical_embedding, colour_embedding, department_embedding,
                                            price_embedding, sales_channel_embedding, season_embedding, day_embedding, month_embbeding, year_embedding,
                                            age_embedding, club_membership_embedding, fashion_news_embedding, postal_code_embedding), dim = 1).to(device)
        output = torch.matmul((customer_embedding_final), torch.t(product_embedding_final)).to(device)
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


# Create function for reading the two preprocessed/transformed transactiosn df datasets creating by enriching for customers and articles
def ReadData(product, customer, features, batch_size, Subset = False):
    prod_features= copy.deepcopy(features)
    customer_features = copy.deepcopy(features)
    customer_features.insert(0, customer)
    prod_features.insert(0, product)
    if(Subset == True):
        UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts_subset.csv')[prod_features]

        Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame_subset.csv')[customer_features]
        Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame_subset.csv')[prod_features]

        if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
            print('There is dimesion error in the data used for the feed forward (model input)')

        splitrange = round(0.75*len(Customer_Preprocessed_data['customer_id']))
        splitrange2 = round(0.95*len(Customer_Preprocessed_data['customer_id']))
        train_customer = Customer_Preprocessed_data.iloc[:splitrange]
        valid_customer = Customer_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_customer = Customer_Preprocessed_data.iloc[splitrange2:]

        train_product = Product_Preprocessed_data.iloc[:splitrange]
        valid_product = Product_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_product = Product_Preprocessed_data.iloc[splitrange2:]
    else:
        UniqueProducts_df = pd.read_csv('Data/Preprocessed/FinalProductDataFrameUniqueProducts_subset.csv')[prod_features]

        Customer_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalCustomerDataFrame_subset.csv')[customer_features]
        Product_Preprocessed_data = pd.read_csv('Data/Preprocessed/FinalProductDataFrame_subset.csv')[prod_features]

        if(Customer_Preprocessed_data.shape != Product_Preprocessed_data.shape):
            print('There is dimesion error in the data used for the feed forward (model input)')

        splitrange = round(0.75*len(Customer_Preprocessed_data['customer_id']))
        splitrange2 = round(0.95*len(Customer_Preprocessed_data['customer_id']))
        train_customer = Customer_Preprocessed_data.iloc[:splitrange]
        valid_customer = Customer_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_customer = Customer_Preprocessed_data.iloc[splitrange2:]

        train_product = Product_Preprocessed_data.iloc[:splitrange]
        valid_product = Product_Preprocessed_data.iloc[splitrange+1:splitrange2]
        test_product = Product_Preprocessed_data.iloc[splitrange2:]

    with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)

    #Customer_data_tensor = torch.tensor(Only_Customer_data[['customer_id','price','age','colour_group_name','department_name']].to_numpy(), dtype = torch.int)
    Product_data_tensor = torch.tensor(UniqueProducts_df.fillna(0).to_numpy(), dtype = torch.int)
    customer_train_tensor = torch.tensor(train_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_train_tensor = torch.tensor(train_product.fillna(0).to_numpy(), dtype = torch.int)

    customer_valid_tensor = torch.tensor(valid_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_valid_tensor = torch.tensor(valid_product.fillna(0).to_numpy(), dtype = torch.int)
    #customer_dataset = CreateDataset(Customer_data_tensor)#,features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_dataset = CreateDataset(Product_data_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])

    customer_test_tensor = torch.tensor(test_customer.fillna(0).to_numpy(), dtype = torch.int)
    product_test_tensor = torch.tensor(test_product.fillna(0).to_numpy(), dtype = torch.int)

    product_train_dataset = CreateDataset(product_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_train_dataset = CreateDataset(customer_train_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_valid_dataset = CreateDataset(product_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_valid_dataset = CreateDataset(customer_valid_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
    product_test_dataset = CreateDataset(product_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['article_id'])
    customer_test_dataset = CreateDataset(customer_test_tensor)#, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])

    dataset_shapes = {'train_shape':product_train_tensor.shape,'valid_shape':product_valid_tensor.shape,'test_shape':product_test_tensor.shape}

    #Training in batches:
    product_train_loader = torch.utils.data.DataLoader(product_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)
    customer_train_loader = torch.utils.data.DataLoader(customer_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    product_valid_loader = torch.utils.data.DataLoader(product_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
    customer_valid_loader = torch.utils.data.DataLoader(customer_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    product_test_loader = torch.utils.data.DataLoader(product_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
    customer_test_loader = torch.utils.data.DataLoader(customer_test_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

    return product_dataset, product_train_loader, customer_train_loader, product_valid_loader, customer_valid_loader, number_uniques_dict, dataset_shapes, product_test_loader, customer_test_loader

batch_size = 2048
## Call the load data function
product_dataset, product_train_loader, customer_train_loader, product_valid_loader, customer_valid_loader, number_uniques_dict, dataset_shapes,_ ,_ = ReadData(
                                                            product='article_id', customer='customer_id',features= ['club_member_status',
                                                            'fashion_news_frequency', 'age', 'postal_code', 'price',
                                                            'sales_channel_id', 'season', 'day', 'month', 'year', 'prod_name',
                                                            'graphical_appearance_name', 'colour_group_name',
                                                            'department_name'], batch_size=batch_size, Subset= True)

embedding_dim = 32
#Intitalize model
model = RecSysModel(product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_unique_dict=number_uniques_dict, device=device, n_ages = 111)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr = 0.001)
model =model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 2


dataiter = iter(product_train_loader)
dataset = next(dataiter)

# Train 2 epochs
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
    for i, product_data_batch,customer_data_batch in zip(np.arange(1,dataset_shapes['train_shape'][0]),product_train_loader,customer_train_loader):
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
        product_id_one_hot = F.one_hot(product_id,num_classes = (number_uniques_dict["n_products"]-1))
        loss = loss_fn(output,product_id_one_hot.type(torch.FloatTensor).to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        epoch_loss.append(loss.item())
        if(i % 100 == 0):
            print(' Train batch {} loss: {}'.format(i, np.mean(epoch_loss)))
        
        epoch_loss.append(loss.item())
    epoch_loss_value = np.mean(epoch_loss)
    Loss_list.append(epoch_loss_value)
    model.eval()
    epoch_valid_loss = []
    for batch, product_data_batch_valid, customer_data_batch_valid in zip(np.arange(1,dataset_shapes['valid_shape'][0]), product_valid_loader, customer_valid_loader):
        product_id = product_data_batch_valid[:,0].type(torch.long)
        outputs = model(customer_data_batch_valid, product_data_batch_valid)
        output = torch.squeeze(outputs, 1)
        product_id_one_hot = F.one_hot(product_id,num_classes = (number_uniques_dict["n_products"]-1))
        loss = loss_fn(output,product_id_one_hot.type(torch.FloatTensor).to(device))
        epoch_valid_loss.append(loss.item())
        if(batch % 100 == 0):
            print(' Valid batch {} loss: {}'.format(batch, np.mean(epoch_valid_loss)))
    epoch_valid_loss_value = np.mean(epoch_valid_loss)
    Valid_Loss_list.append(epoch_valid_loss_value)
    if(epoch_valid_loss_value < Best_loss):
        best_model = copy.deepcopy(model)
        Best_loss = epoch_valid_loss_value
PATH = 'Models/Baseline_MulitDim_model.pth'
#torch.save(best_model, PATH)
torch.save(model.state_dict(), PATH)


#prof.stop()


print("finished training")
print("Loss list = ", Loss_list)

plt.plot(np.arange(1,len(Loss_list)+1), Loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training graph')
plt.show()
