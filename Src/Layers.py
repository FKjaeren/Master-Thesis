
import numpy as np
from torch import nn
import torch

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, num_fields, n_unique_dict,device,n_ages=111):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_unique_dict = n_unique_dict
        self.n_ages = n_ages
        self.num_fields = len(num_fields)-1
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
    def forward(self, x):
        customer_embedding_final = torch.zeros((x.shape[0], self.num_fields, self.embedding_dim))
        customer_embedding_final[:,0,:] = self.customer_embedding(x[:,0])
        customer_embedding_final[:,1,:] = self.product_embedding(x[:,1])
        customer_embedding_final[:,2,:] = self.price_embedding(x[:,2])
        customer_embedding_final[:,3,:] = self.sales_channel_id_embedding(x[:,3])
        customer_embedding_final[:,4,:] = self.day_embedding(x[:,4])
        customer_embedding_final[:,5,:] = self.month_embedding(x[:,5])
        customer_embedding_final[:,6,:] = self.year_embedding(x[:,6])
        customer_embedding_final[:,7,:] = self.season_embedding(x[:,7])
        customer_embedding_final[:,8,:] = self.prod_name_embedding(x[:,8])
        customer_embedding_final[:,9,:] = self.graphical_embedding(x[:,9])
        customer_embedding_final[:,10,:] = self.colour_embedding(x[:,10])
        customer_embedding_final[:,11,:] = self.department_embedding(x[:,11])
        customer_embedding_final[:,12,:] = self.club_member_status_embedding(x[:,12])
        customer_embedding_final[:,13,:] = self.fashion_news_frequency_embedding(x[:,13])
        customer_embedding_final[:,14,:] = self.age_embedding(x[:,14])
        customer_embedding_final[:,15,:] = self.postal_code_embedding(x[:,15])

        return customer_embedding_final

class MultiLayerPerceptron(torch.nn.Module):
    """
    Class to instantiate a Multilayer Perceptron model
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)



class FactorizationMachine(torch.nn.Module):
    """
        Class to instantiate a Factorization Machine model
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``


        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class LinearLayer(torch.nn.Module):
    def __init__(self, output_dim = 1):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
    def forward(self,x):
        return torch.sum(x, dim=1) + self.bias