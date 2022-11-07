import pandas as pd
import torch
import numpy as np
import random
import pickle

def CreateNegativeSamples(df, train_df, num_negative_samples, type_df = 'Train', method = 'Random_choices', customer_id = None, article_df=None, customer_df = None, batch_size = None):
    if(method == 'Random_choices'):
        if(type_df == 'Train'):
            unique_train_customers = df.customer_id.unique()
            unique_train_articles = df.article_id.unique()
            if(num_negative_samples == len(unique_train_articles)):
                interactions_list = []
                for c in unique_train_customers:
                    for item in range(unique_train_articles):
                        interactions_list.append([c,item,0])
            else:
                interactions_list = []
                for c in unique_train_customers:
                    for i in range(num_negative_samples):
                        item = random.choice(unique_train_articles)
                        interactions_list.append([c,item,0])


            map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}


            negative_df = pd.DataFrame(data = interactions_list, columns = ['customer_id','article_id','negative_values'])
            negative_df['day'] = np.random.randint(1, 28, negative_df.shape[0])
            negative_df['month'] = np.random.randint(1, 12, negative_df.shape[0])
            #negative_df['year'] = np.random.randint(0, transactions_df.year.unique().max(), negative_df.shape[0])
            negative_df['year'] = np.zeros(negative_df.shape[0])

            negative_df.loc[(negative_df['month']>= 1) & (negative_df['month'] <=2), 'season'] = 'Winter'
            negative_df.loc[(negative_df['month'] == 12), 'season'] = 'Winter' 
            negative_df.loc[(negative_df['month'] >= 3) & (negative_df['month'] <=5), 'season'] = 'Spring' 
            negative_df.loc[(negative_df['month'] >= 6) & (negative_df['month'] <=8),'season'] = 'Summer' 
            negative_df.loc[(negative_df['month'] >= 9) & (negative_df['month'] <=11), 'season'] = 'Autumn' 
            negative_df['season'] = negative_df['season'].map(map_season)


            temp = train_df[['article_id','price','sales_channel_id']].groupby(['article_id']).mean().reset_index()
            temp['sales_channel_id'] = temp['sales_channel_id'].round()
            temp['price'] = temp['price'].round(decimals=4)

            negative_df = negative_df.merge(temp, on = 'article_id', how = 'left')
            print('Negative samples were created for the train dataframe, with the method "Random Choices"')
        elif(type_df == 'Test'):
            unique_train_customers = df.customer_id.unique()
            article_df = pd.read_csv('Data/Raw/articles.csv')
            with open('Models/Article_Id_Encoder.sav', "rb") as input_file: 
                Article_id_encoder = pickle.load(input_file)
            article_df['article_id'] = Article_id_encoder.transform(article_df['article_id'].to_numpy().reshape(-1, 1))
            unique_train_articles = article_df.article_id.unique()
            map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}

            num_negative = num_negative_samples
            interactions_list = []
            for c in unique_train_customers:
                for i in range(num_negative):
                    item = unique_train_articles[i]
                    interactions_list.append([c,item,0])

            negative_df = pd.DataFrame(data = interactions_list, columns = ['customer_id','article_id','negative_values'])
            negative_df['day'] = np.random.randint(1, 28, negative_df.shape[0])
            negative_df['month'] = np.random.randint(1, 12, negative_df.shape[0])
            #negative_df['year'] = np.random.randint(0, transactions_df.year.unique().max(), negative_df.shape[0])
            negative_df['year'] = np.zeros(negative_df.shape[0])

            negative_df.loc[(negative_df['month']>= 1) & (negative_df['month'] <=2), 'season'] = 'Winter'
            negative_df.loc[(negative_df['month'] == 12), 'season'] = 'Winter' 
            negative_df.loc[(negative_df['month'] >= 3) & (negative_df['month'] <=5), 'season'] = 'Spring' 
            negative_df.loc[(negative_df['month'] >= 6) & (negative_df['month'] <=8),'season'] = 'Summer' 
            negative_df.loc[(negative_df['month'] >= 9) & (negative_df['month'] <=11), 'season'] = 'Autumn' 
            negative_df['season'] = negative_df['season'].map(map_season)


            temp = train_df[['article_id','price','sales_channel_id']].groupby(['article_id']).mean().reset_index()
            temp['sales_channel_id'] = temp['sales_channel_id'].round()
            temp['price'] = temp['price'].round(decimals=4)

            negative_df = negative_df.merge(temp, on = 'article_id', how = 'left')
            print('Negative samples were created for the test dataframe, with the method "Random Choices"')
        else:
            print('Unreconized requested dataframe type, with the method "Random Choices"')
    elif(method == 'Bayesian_sampling'):
        print('TODO du får nada')
    elif(method == 'OneCustomerNegSamples'):
        #unique_train_customers = df.customer_id.unique()  
        interactions_list = []
        unique_train_articles = article_df.article_id.unique()
        for c in customer_id:
            for i in range(num_negative_samples):
                item = random.choice(unique_train_articles)
                interactions_list.append([c.item(),item,0])
        map_season = {'Winter': 0, 'Spring':1, 'Summer': 2, 'Autumn': 3}
        negative_df = pd.DataFrame(data = interactions_list, columns = ['customer_id','article_id','negative_values'])
        negative_df['day'] = np.random.randint(1, 28, negative_df.shape[0])
        negative_df['month'] = np.random.randint(1, 12, negative_df.shape[0])
        #negative_df['year'] = np.random.randint(0, transactions_df.year.unique().max(), negative_df.shape[0])
        negative_df['year'] = np.zeros(negative_df.shape[0])

        negative_df.loc[(negative_df['month']>= 1) & (negative_df['month'] <=2), 'season'] = 'Winter'

        negative_df.loc[(negative_df['month'] == 12), 'season'] = 'Winter' 
        negative_df.loc[(negative_df['month'] >= 3) & (negative_df['month'] <=5), 'season'] = 'Spring' 
        negative_df.loc[(negative_df['month'] >= 6) & (negative_df['month'] <=8),'season'] = 'Summer' 
        negative_df.loc[(negative_df['month'] >= 9) & (negative_df['month'] <=11), 'season'] = 'Autumn' 
        negative_df['season'] = negative_df['season'].map(map_season)

        temp = pd.DataFrame(data = train_df[:,1:4].view(batch_size,3).numpy(), columns = ['article_id','price', 'sales_channel_id'])
        temp = temp.groupby(['article_id']).mean().reset_index()
        temp['sales_channel_id'] = temp['sales_channel_id'].round()
        temp['price'] = temp['price'].round(decimals=4)

        negative_df = negative_df.merge(temp, on = 'article_id', how = 'left')
        negative_df = negative_df.merge(article_df, on = 'article_id', how = 'left')
        negative_df = negative_df.merge(customer_df, how = 'left', on ='customer_id')
        negative_df = negative_df[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
            'month', 'year', 'season', 'prod_name', 'product_type_name',
            'graphical_appearance_name', 'colour_group_name', 'department_name',
            'index_group_name', 'FN', 'Active', 'club_member_status',
            'fashion_news_frequency', 'age', 'postal_code','negative_values']]
        negative_df = torch.tensor(negative_df.fillna(0).to_numpy(), dtype = torch.int)
        negative_df = negative_df.view(batch_size,num_negative_samples,21)
        #print('Negative samples were created for the train dataframe, with the method "Random Choices"')
    else:
        print('Unreconized method')
    return negative_df