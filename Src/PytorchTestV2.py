import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing, metrics
import numpy as np
import copy

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
    def __init__(self, Customer_id, Article_id, embedding_dim):
        super().__init__()
        self.num_customers = len(Customer_id)
        self.num_articles = len(Article_id)
        self.all_Articles = torch.from_numpy(Article_id).type(torch.int64)
        self.customer_embedding = nn.Embedding(self.num_customers, embedding_dim)
        self.article_embedding = nn.Embedding(self.num_articles, embedding_dim)
        #self.dot = torch.dot()
        self.out = nn.Linear(64,5)

    def monitor_metrics(self, output, target):
        output = output.detach.cpu().numpy()
        target = target.detach.cpu().numpy()
        return {'rmse':np.sqrt(metrics.mean_squared_error(target, output))}

    def forward(self, customer, article):
        customer_embedding = self.customer_embedding(customer)
        article_embedding = self.article_embedding(article)
        #output = torch.cat([customer_embedding, article_embedding], dim = 1)

        output = torch.dot(customer_embedding, article_embedding)
        calc_metrics = self.monitor_metrics(output,article.view(1,-1))
        return output, calc_metrics

    def TrainModel(self, customer, k):
        customer_embedding = self.customer_embedding(customer)
        all_articles_embedding = self.article_embedding(self.all_Articles)
        matrixfactorization = torch.matmul(torch.t(customer_embedding), torch.t(all_articles_embedding))
        return matrixfactorization, indexes

    def CustomerItemRecommendation(self, customer, k):
        customer_embedding = self.customer_embedding(customer)
        all_articles_embedding = self.article_embedding(self.all_Articles)

        matrixfactorization = torch.matmul(torch.t(customer_embedding), torch.t(all_articles_embedding))
        recommendations, indexes = torch.topk(matrixfactorization, k = k)
        return recommendations, indexes


train_df = pd.read_csv('Data/Preprocessed/TrainData.csv')
train_df = train_df.iloc[0:100000]
valid_df = pd.read_csv('Data/Preprocessed/ValidData.csv')
test_df = pd.read_csv('Data/Preprocessed/TestData.csv')

Customer_id = pd.read_csv('Data/Raw/customers.csv')['customer_id'].values.flatten()
Article_id = pd.read_csv('Data/Raw/articles.csv')['article_id'].values

Customer_id_Encoder = preprocessing.LabelEncoder().fit(Customer_id)
Article_id_Encoder = preprocessing.LabelEncoder().fit(Article_id)
Customer_id_Encoded = Customer_id_Encoder.transform(Customer_id)
Article_id_Encoded = Article_id_Encoder.transform(Article_id)

train_dataset = copy.deepcopy(train_df)
train_dataset.customer_id = Customer_id_Encoder.transform(train_df.customer_id.values)
train_dataset.article_id = Article_id_Encoder.transform(train_df.article_id.values)
valid_dataset = copy.deepcopy(valid_df)
valid_dataset.customer_id = Customer_id_Encoder.transform(valid_df.customer_id.values)
valid_dataset.article_id = Article_id_Encoder.transform(valid_df.article_id.values)

#processed_train = dataset_test(train_df['customer_id'], train_df['article_id'])

model = RecSysModel(Customer_id_Encoded, Article_id_Encoded, embedding_dim=64)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)


train_dataset = CreateDataset(train_dataset['customer_id'],train_dataset['article_id'])
valid_dataset = CreateDataset(valid_dataset['customer_id'], valid_dataset['article_id'])



loss_fn = torch.nn.CrossEntropyLoss()
num_epochs = 5

"""
Training in batches:
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, num_workers = 2, shuffle = True)
valid_laoder = torch.utils.data.DataLoader(valid_dataset, batch_size = 32, num_workers = 2, shuffle = True)

dataiter = iter(train_loader)
customer_id, article_id = dataiter.next()

for epoch in range(1,num_epochs):
    print(epoch)
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        print(data)
        inputs, labels = data.items()
        #print(inputs, labels)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model.TrainModel(inputs, k)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / len(train_loader) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

"""

for epoch in range(1,num_epochs):
    print(epoch)
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    # Every data instance is an input + label pair
    for i in range(0,len(train_dataset)):
        inputs, labels = train_dataset[i]
        #print(inputs, labels)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, indexes = model.TrainModel(inputs,1)

        # Compute the loss and its gradients
        labels_one_hot = torch.zeros(len(Article_id))
        labels_one_hot[labels] = 1
        loss = loss_fn(outputs, labels_one_hot)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

            # Gather data and report
        running_loss += loss.item()
        if(i % 100 == 0):
            last_loss = running_loss / len(train_dataset) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            #tb_x = epoch * len(train_loader) + i + 1
            #tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    running_loss = 0.


print("finished training")