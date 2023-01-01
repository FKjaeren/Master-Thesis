from torch import nn
import torch
from sklearn import metrics
import numpy as np
## Create MF multiclass model architeture
class RecSysModel(nn.Module):
    def __init__(self, Products_data, embedding_dim, batch_size, n_unique_dict,device,n_ages=111):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_unique_dict = n_unique_dict
        self.n_ages = n_ages
        ## Define embedding layer for every class
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

        #self.out = nn.Linear(64,n_products+1)

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