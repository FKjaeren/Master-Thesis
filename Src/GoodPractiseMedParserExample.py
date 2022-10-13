from tkinter import E
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
from evaluate import evaluate_model
from utils import train_one_epoch, test, plot_statistics
torch.manual_seed(0)

from Dataset import MovieLensDataset
# Python imports
import argparse
from time import time
import numpy as np
import pickle

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='movielens',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[16,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help="Regularization for each layer")
    parser.add_argument('--num_neg_train', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance while training')
    parser.add_argument('--num_neg_test', type=int, default=100,
                        help='Number of negative instances to pair with a positive instance while testing')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Add dropout layer after each dense layer, with p = dropout_prob')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


class MLP(nn.Module):

    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert (layers[0] % 2 == 0), "layers[0] must be an even number"
        self.__alias__ = "MLP {}".format(layers)
        self.__dropout__ = dropout

        # user and item embedding layers
        embedding_dim = int(layers[0]/2)
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)

        # list of weight matrices
        self.fc_layers = torch.nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x,  p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(
                    feed_dict[key]).to(dtype=torch.long, device=device)
        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__


class Mapper():
    def __init__(self, possible_articles, num_negative_articles):
        self.num_possible_articles = len(possible_articles)
        self.possible_articles_tensor = torch.tensor(possible_articles)
        self.num_negative_articles = num_negative_articles
        self.y = torch.nn.functional.one_hot(self.possible_articles_tensor, -1)
    def __call__(self, customer, article):
        random_negatives_indexes  = torch.randint(0,self.num_possible_articles,self.num_negative_articles)
        negative_products =  self.possible_articles_tensor[random_negatives_indexes]
        candidates = torch.concat([article, negative_products], axis = 0)
        return (customer, candidates)#, self.y

def get_dataset(df, articles, number_negative_articles):
    #dummy_customer_tensor = torch.tensor(df[['customer_id']].values)
    #article_tensor = torch.tensor(df['article_id'])
    #df_tensor = torch.cat((torch.tensor(df['Customer_ID']), torch.tensor(df['Article_Id'])), dim = 1)
    df_tensor = torch.tensor ([df['Customer_ID'],df['Article_Id']])
    #dataset = df_tensor.map(Mapper(articles, number_negative_articles))
    num_possible_articles = len(articles)
    possible_articles_tensor = torch.tensor(articles)
    y = torch.cat((torch.reshape(possible_articles_tensor, (len(possible_articles_tensor),1)),torch.zeros((len(possible_articles_tensor),number_negative_articles))), dim = 1)
    random_negatives_indexes  = torch.randint(0,num_possible_articles,number_negative_articles)
    negative_products =  possible_articles_tensor[random_negatives_indexes]
    candidates = torch.concat([df[0][:], negative_products], axis = 0)
    return df_tensor, candidates

class dataset_test:
    def __init__(self, customer_id, article_id):
        self.customer_id = customer_id
        self. article_id = article_id
    def __len__(self):
        return len(self.customer_id)
    def __getitem__(self, article_id):
        customer_id = self.customer_id[article_id]
        article_id = self.article_id[article_id]
        return{"customer_id":torch.tensor(customer_id, dtype = torch.long),
        "article_id":torch.tensor(article_id, dtype = torch.int)}

test = dataset_test(train_df['customer_id'], train_df['article_id'])

def main():
    Customer_id = pd.read_csv('Data/Raw/customers.csv')['customer_id'].values.flatten()
    Article_id = pd.read_csv('Data/Raw/articles.csv')['article_id'].values.flatten()
    Customer_id_Encoder = preprocessing.LabelEncoder().fit(Customer_id)
    Article_id_Encoder = preprocessing.LabelEncoder().fit(Article_id)
    Customer_id_Encoded = Customer_id_Encoder.transform(Customer_id)
    Article_id_Encoded = Article_id_Encoder.transform(Article_id)
    all_users_items = {'Customer_ID' : Customer_id_Encoded, 'Article_Id':Article_id_Encoded}

    num_users = len(all_users_items['Customer_ID'])
    num_items = len(all_users_items['Article_Id'])

    train_df = pd.read_csv('Data/Preprocessed/TrainData.csv')
    valid_df = pd.read_csv('Data/Preprocessed/ValidData.csv')
    test_df = pd.read_csv('Data/Preprocessed/TestData.csv')



    feed_dict = {'Customer_ID' :  Customer_id_Encoder.transform(train_df['customer_id']), 'Article_Id': Article_id_Encoder.transform(train_df['article_id'])}

    full_dataset = get_dataset(feed_dict, all_users_items['Article_Id'], 5)

    args = parse_args()
    path = args.path
    #dataset = args.dataset
    layers = eval(args.layers)
    weight_decay = args.weight_decay
    num_negatives_train = args.num_neg_train
    num_negatives_test = args.num_neg_test
    dropout = args.dropout
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    print("MLP arguments: %s " % (args))
    # model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())

    # Load data

    t1 = time()
    dataset = 'movielens'
    path = '/Data/Raw/ml-latest-small/'
    full_dataset = MovieLensDataset(path + dataset, num_negatives_train=num_negatives_train, num_negatives_test=num_negatives_test)
    train, testRatings, testNegatives = full_dataset.trainMatrix, full_dataset.testRatings, full_dataset.testNegatives
    #num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    training_data_generator = DataLoader(
        full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Build model
    model = MLP(num_users, num_items, layers=layers, dropout=dropout)
    # Transfer the model to GPU, if one is available
    model.to(device)
    if verbose:
        print(model)

    loss_fn = torch.nn.BCELoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)

    # Record performance
    hr_list = []
    ndcg_list = []
    BCE_loss_list = []

    # Check Init performance
    hr, ndcg = test(model, full_dataset, topK)
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    BCE_loss_list.append(1)
    # do the epochs now

    for epoch in range(epochs):
        epoch_loss = train_one_epoch( model, training_data_generator, loss_fn, optimizer, epoch, device)

        if epoch % verbose == 0:
            hr, ndcg = test(model, full_dataset, topK)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
            BCE_loss_list.append(epoch_loss)
            # if hr > best_hr:
            #     best_hr, best_ndcg, best_iter = hr, ndcg, epoch
            #     if args.out > 0:
            #         model.save(model_out_file, overwrite=True)
    print("hr for epochs: ", hr_list)
    print("ndcg for epochs: ", ndcg_list)
    print("loss for epochs: ", BCE_loss_list)
    # plot_statistics(hr_list, ndcg_list, BCE_loss_list,model.get_alias(), "./figs")
    # with open("metrics", 'wb') as fp:
    #     pickle.dump(hr_list, fp)
    #     pickle.dump(ndcg_list, fp)

    best_iter = np.argmax(np.array(hr_list))
    best_hr = hr_list[best_iter]
    best_ndcg = ndcg_list[best_iter]
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %
          (best_iter, best_hr, best_ndcg))
    # if args.out > 0:
    #     print("The best MLP model is saved to %s" %(model_out_file))


if __name__ == "__main__":
    print("Device available: {}".format(device))
    main()