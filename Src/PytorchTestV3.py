import torch
import torchvision
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

class CreateDataset(Dataset):
    def __init__(self, dataset, features, idx_variable):
        #self.customer_id = dataset['customer_id']
        #self.product_id = dataset['prod_name']
        self.id = idx_variable
        self.features = features
        self.all_data = dataset

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, transaction):
        #customer_id = self.customer_id[transaction]
        #product_id = self.product_id[transaction]
        features = torch.tensor(self.all_data[self.features].to_numpy(), dtype = torch.int)
        idx_variable = torch.tensor(self.all_data[self.id].to_numpy(), dtype = torch.int)
        all_data = torch.cat((idx_variable, features), dim = 1)
        #data_tensor = torch.tensor(self.all_data.to_numpy(), dtype = torch.int)
        return all_data[transaction]
    def shape(self):
        shape_value = self.all_data.shape
        return shape_value

class RecSysModel(torch.nn.Module):
    def __init__(self, Customer_data, Products_data, embedding_dim, batch_size, n_products, n_customers, n_prices, n_colours, n_departments,n_ages=111):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_customers = n_customers
        self.num_products = n_products
        self.num_prices = n_prices
        self.num_ages = n_ages
        self.num_colours = n_colours
        self.num_departments = n_departments
        self.customer_embedding = nn.Embedding(self.num_customers+2, embedding_dim)        
        self.product_embedding = nn.Embedding(self.num_products+2, embedding_dim)
        self.price_embedding = nn.Embedding(self.num_prices+2, embedding_dim)
        self.age_embedding = nn.Embedding(self.num_ages+2,embedding_dim)
        self.colour_embedding = nn.Embedding(self.num_colours+2, embedding_dim)
        self.department_embedding = nn.Embedding(self.num_departments+2, embedding_dim)

        self.All_Products = Products_data

        #self.dot = torch.dot()
        self.out = nn.Linear(64,n_products+1)

    def monitor_metrics(self, output, target):
        output = output.detatch().numpy()
        target = target.detatch().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, Customer_data, Product_data):
        customer_embedding = self.customer_embedding(Customer_data[:,0])
        price_embedding = self.price_embedding(Customer_data[:,1])
        age_embedding = self.age_embedding(Customer_data[:,2])
        colour_embedding = self.colour_embedding(Customer_data[:,3])
        department_embedding = self.department_embedding(Customer_data[:,4])
        customer_embedding_final = torch.cat((customer_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1)
        #for i in range(1,Customer_data.shape[1]):
        #    print('feature i= ',i)
        #    customer_embedding_temp = self.customer_embedding(Customer_data[:,i].reshape(-1,1))
        #    customer_embedding = torch.cat((customer_embedding,customer_embedding_temp), dim = 1)
        product_embedding = self.product_embedding(Product_data[:,0])
        price_embedding = self.price_embedding(Product_data[:,1])
        age_embedding = self.age_embedding(Product_data[:,2])
        colour_embedding = self.colour_embedding(Product_data[:,3])
        department_embedding = self.department_embedding(Product_data[:,4])
        product_embedding_final = torch.cat((product_embedding, price_embedding, age_embedding, colour_embedding, department_embedding), dim = 1)
        #for i in range(1,Product_data.shape[1]):
        #    print("product i: ",i)
        #    product_embedding_temp = self.product_embedding(Product_data[:,i].reshape(-1,1))
        #    product_embedding = torch.cat((product_embedding,product_embedding_temp), dim = 1)
        print('test1')
        output = torch.matmul((product_embedding_final), torch.t(customer_embedding_final))
        print('test2')
        output = self.out(output)
        #output = output.long()
        #calc_metrics = self.monitor_metrics(output,Product_data[:,0].view(1,-1))
        return output#, calc_metrics

    def TrainModel(self, Customer_data, Product_data):
        customer_embedding = self.customer_embedding(Customer_data[:][0])
        for i in range(1,Customer_data.shape[1]):
            customer_embedding_temp = customer_embedding(Customer_data[:][i])
            customer_embedding = torch.cat((customer_embedding,customer_embedding_temp), dim = 1)
        product_columns = self.All_Products.columns
        all_products_embedding = self.product_embedding(self.All_Products[product_columns[0]])
        for i in range(1,len(product_columns)):
            all_products_embedding_temp = self.product_embedding(self.All_Products[product_columns[i]])
            all_products_embedding = torch.cat((all_products_embedding,all_products_embedding_temp), dim = 1)

        product_embedding = self.product_embedding(Product_data[:][0])
        for i in range(1,Customer_data.shape[1]):
            product_embedding_temp = product_embedding(Product_data[:][i])
            product_embedding = torch.cat((product_embedding,product_embedding_temp), dim = 1)
        matrixfactorization = torch.matmul(torch.t(product_embedding).reshape(self.batch_size,1,embedding_dim), torch.t(product_embedding))
        return matrixfactorization

    def CustomerItemRecommendation(self, customer, k):
        customer_embedding = self.customer_embedding(customer)
        all_products_embedding = self.product_embedding(self.All_Products)

        matrixfactorization = torch.mm(torch.t(customer_embedding), torch.t(all_products_embedding))
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
num_products = all_products['prod_name'].nunique()
num_departments = all_products['department_name'].nunique()
num_colours = all_products['colour_group_name'].nunique()
max_age = 110


Customer_id_Encoder = preprocessing.LabelEncoder().fit(all_customers['customer_id'])
Product_Encoder = preprocessing.LabelEncoder().fit(all_products['prod_name'])
Customer_data['customer_id'] = Customer_id_Encoder.transform(Customer_data['customer_id'])
Product_data['product_id'] = Product_Encoder.transform(Product_data['prod_name'])
Product_data = Product_data.drop(columns=['prod_name'], axis = 1)


Colour_Encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_colours+1).fit(Product_data[['colour_group_name']].to_numpy().reshape(-1, 1))
Department_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_departments+1).fit(Product_data[['department_name']].to_numpy().reshape(-1, 1))
Age_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=110).fit(all_customers[['age']].to_numpy().reshape(-1, 1))

all_prices = np.concatenate((Product_data['price'].unique(),Customer_data['price'].unique()),axis = 0)
all_prices = np.unique(all_prices)
num_prices = len(all_prices)

Price_encoder = preprocessing.OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=num_prices+1).fit(all_prices.reshape(-1, 1))


train_dataset = copy.deepcopy(train_df)
train_dataset['price'] = train_dataset['price'].round(decimals=4)
train_dataset.customer_id = Customer_id_Encoder.transform(train_df.customer_id.values)
train_dataset['product_id'] = Product_Encoder.transform(train_df.prod_name.values)
train_dataset['colour_group_name'] = Colour_Encoder.transform(train_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
train_dataset['department_name'] = Department_encoder.transform(train_dataset[['department_name']].to_numpy().reshape(-1, 1))
train_dataset['age'] = Age_encoder.transform(train_dataset[['age']].to_numpy().reshape(-1,1))
train_dataset['price'] = Price_encoder.transform(train_dataset[['price']].to_numpy().reshape(-1,1)).astype(int)

valid_dataset = copy.deepcopy(valid_df)
valid_dataset.customer_id = Customer_id_Encoder.transform(valid_df.customer_id.values)
valid_dataset['product_id'] = Product_Encoder.transform(valid_df.prod_name.values)
valid_dataset['colour_group_name'] = Colour_Encoder.transform(valid_dataset[['colour_group_name']].to_numpy().reshape(-1, 1))
valid_dataset['department_name'] = Department_encoder.transform(valid_dataset[['department_name']].to_numpy().reshape(-1, 1))
valid_dataset['age'] = Age_encoder.transform(valid_dataset[['age']].to_numpy().reshape(-1,1))

Product_data['colour_group_name'] = Colour_Encoder.transform(Product_data[['colour_group_name']].to_numpy().reshape(-1, 1))
Product_data['department_name'] = Department_encoder.transform(Product_data[['department_name']].to_numpy().reshape(-1, 1))
Product_data = Product_data[['product_id','age','price','sales_channel_id','colour_group_name','department_name']]
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

#train_dataset = train_dataset.dropna(subset = ['age'])

#train_dataset = pd.get_dummies(train_dataset, columns = ['prod_name','colour_group_name','department_name'])

customer_dataset = CreateDataset(Customer_data,features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_dataset = CreateDataset(Product_data, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])



product_train_dataset = CreateDataset(train_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])
customer_train_dataset = CreateDataset(train_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])
product_valid_dataset = CreateDataset(valid_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['product_id'])
customer_valid_dataset = CreateDataset(valid_dataset, features=['price','age','colour_group_name','department_name'],idx_variable=['customer_id'])


#processed_train = dataset_test(train_df['customer_id'], train_df['article_id'])
batch_size = 1024
embedding_dim = 64
model = RecSysModel(customer_dataset, product_dataset, embedding_dim=embedding_dim, batch_size=batch_size, n_products=num_products,n_customers=num_customers, n_prices=num_prices, n_colours=num_colours, n_departments=num_departments)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001, lr = 0.005)

loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 3


#Training in batches:
product_train_loader = torch.utils.data.DataLoader(product_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = False, drop_last = True)
customer_train_loader = torch.utils.data.DataLoader(customer_train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

product_valid_loader = torch.utils.data.DataLoader(product_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
customer_valid_loader = torch.utils.data.DataLoader(customer_valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

#Num_classes = len(Product_data['product_id'])
dataiter = iter(product_train_loader)
dataset = dataiter.next()


Loss_list = []

for epoch in range(1,num_epochs):
    running_loss = 0.
    last_loss = 0.
    epoch_loss = []
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

        # Make predictions for this batch
        outputs = model(customer_data_batch, product_data_batch)
        output = torch.squeeze(outputs, 1)
        #labels_one_hot = F.one_hot(product_id, num_classes=Num_classes)
        # Compute the loss and its gradients
        #labels_one_hot = torch.zeros(Num_classes, batch_size)
        #labels_one_hot[product_id] = 1
        print(output)
        print(product_id)
        loss = loss_fn(output,product_id)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        
        #running_loss += loss.item()
        #if(i % 10 == 0):
        #    last_loss = running_loss / (i+1) # loss per batch
        #    print('  batch {} loss: {}'.format(i + 1, last_loss))
        #    #tb_x = epoch * len(train_loader) + i + 1
        #    #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        
        epoch_loss.append(loss.item())
    epoch_loss = np.mean(epoch_loss)
    Loss_list.append(epoch_loss)
    epoch_loss = 0
    running_loss = 0.

print("finished training")
print("Loss list = ", Loss_list)

plt.plot(np.arange(1,len(Loss_list)+1), Loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training graph')
plt.show()