from itertools import product
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
    def __init__(self, customer_id, article_id):
        self.customer_id = customer_id
        self. article_id = article_id

    def __len__(self):
        return len(self.customer_id)

    def __getitem__(self, transaction):
        customer_id = self.customer_id[transaction]
        article_id = self.article_id[transaction]
        #sample = {'customer_id':torch.tensor(customer_id, dtype = torch.long), 'article_id':torch.tensor(article_id, dtype = torch.int)}
        return torch.tensor(customer_id, dtype = torch.long), torch.tensor(article_id, dtype = torch.int)

class RecSysModel(torch.nn.Module):
    def __init__(self, Customer, Products, embedding_dim, batch_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.num_customers = len(Customer)
        self.num_products = len(Products)
        self.All_Products = torch.from_numpy(Products).type(torch.int64)
        self.customer_embedding = nn.Embedding(self.num_customers, embedding_dim)
        self.product_embedding = nn.Embedding(self.num_products, embedding_dim)
        #self.dot = torch.dot()
        self.out = nn.Linear(64,5)

    def monitor_metrics(self, output, target):
        output = output.detach.cpu().numpy()
        target = target.detach.cpu().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, customer, product):
        customer_embedding = self.customer_embedding(customer)
        product_embedding = self.product_embedding(product)
        #output = torch.cat([customer_embedding, article_embedding], dim = 1)

        output = torch.dot(customer_embedding, product_embedding)
        calc_metrics = self.monitor_metrics(output,product.view(1,-1))
        return output, calc_metrics

    def TrainModel(self, customer):
        customer_embedding = self.customer_embedding(customer)
        all_products_embedding = self.product_embedding(self.All_Products)
        matrixfactorization = torch.matmul(torch.t(customer_embedding).reshape(self.batch_size,1,embedding_dim), torch.t(all_products_embedding))
        return matrixfactorization

    def CustomerItemRecommendation(self, customer, k):
        customer_embedding = self.customer_embedding(customer)
        all_products_embedding = self.product_embedding(self.All_Products)

        matrixfactorization = torch.mm(torch.t(customer_embedding), torch.t(all_products_embedding))
        recommendations, indexes = torch.topk(matrixfactorization, k = k)
        return recommendations, indexes


train_df = pd.read_csv('Data/Preprocessed/TrainData.csv')
train_df = train_df.iloc[0:100000]
valid_df = pd.read_csv('Data/Preprocessed/ValidData.csv')
test_df = pd.read_csv('Data/Preprocessed/TestData.csv')

Customer_id = pd.read_csv('Data/Raw/customers.csv')['customer_id'].values.flatten()
Products = pd.read_csv('Data/Raw/articles.csv')['prod_name'].values

Customer_id_Encoder = preprocessing.LabelEncoder().fit(Customer_id)
Product_Encoder = preprocessing.LabelEncoder().fit(Products)
Customer_id_Encoded = Customer_id_Encoder.transform(Customer_id)
Product_id = Product_Encoder.transform(Products)

train_dataset = copy.deepcopy(train_df)
train_dataset.customer_id = Customer_id_Encoder.transform(train_df.customer_id.values)
train_dataset['product_id'] = Product_Encoder.transform(train_df.prod_name.values)
valid_dataset = copy.deepcopy(valid_df)
valid_dataset.customer_id = Customer_id_Encoder.transform(valid_df.customer_id.values)
valid_dataset['product_id'] = Product_Encoder.transform(valid_df.prod_name.values)

#processed_train = dataset_test(train_df['customer_id'], train_df['article_id'])
batch_size = 1024
embedding_dim = 64
model = RecSysModel(Customer_id_Encoded, Product_id, embedding_dim=embedding_dim, batch_size=batch_size)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001, lr = 0.005)


train_dataset = CreateDataset(train_dataset['customer_id'],train_dataset['product_id'])
valid_dataset = CreateDataset(valid_dataset['customer_id'], valid_dataset['product_id'])



loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 3


#Training in batches:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)
valid_laoder = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, num_workers = 0, shuffle = True, drop_last = True)

Num_classes = len(Product_id)
dataiter = iter(train_loader)
customer_id, product_id = dataiter.next()


Loss_list = []

for epoch in range(1,num_epochs):
    running_loss = 0.
    last_loss = 0.
    epoch_loss = []
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # Every data instance is an input + label pair
    for i, data in enumerate(train_loader):
        customer_id, product_id = data
        #product_id = product_id.view(batch_size,1)
        product_id = product_id.type(torch.LongTensor)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model.TrainModel(customer_id)
        output = torch.squeeze(outputs, 1)
        #labels_one_hot = F.one_hot(product_id, num_classes=Num_classes)
        # Compute the loss and its gradients
        #labels_one_hot = torch.zeros(Num_classes, batch_size)
        #labels_one_hot[product_id] = 1
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