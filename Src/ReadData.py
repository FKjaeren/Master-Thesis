from posixpath import split
import pandas as pd
import numpy as np

data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')

transactions_df['date'] = pd.to_datetime(transactions_df['t_dat'])

## 80/20 train test split.

splitrange = round(0.8*len(transactions_df['date']))

transactions_train = transactions_df.iloc[:splitrange]
transactions_test = transactions_df.iloc[splitrange+1:]

### Create age intervals on customers:

customer_df['age_interval'] = pd.cut(customer_df['age'],5,right=False)

def encode_fashion_news(customer):
    if(customer == 'None' or customer == 'NONE'):
        encoded_news_frequency = 0
    elif(customer == 'Monthly'):
        encoded_news_frequency = 1
    elif(customer == 'Regularly'):
        encoded_news_frequency = 2
    elif(customer != customer):
        encoded_news_frequency = 0
    return encoded_news_frequency

customer_df['encoded_news_frequency'] = customer_df['fashion_news_frequency'].apply(encode_fashion_news)

def encode_club_membership_status(customer_status):
    if(customer_status == 'LEFT CLUB'):
        encoded_status = 0
    elif(customer_status == 'PRE-CREATE'):
        encoded_status = 1
    elif(customer_status == 'ACTIVE'):
        encoded_status = 2
    elif(customer_status != customer_status):
        encoded_status = 0
    return encoded_status

customer_df['encoded_customer_status'] = customer_df['club_member_status'].apply(encode_club_membership_status)

customer_df['products_shopped']
## Customer item table:







### Next we will combine the different dataset:

X_train = transactions_train.merge(customer_df,how ='left', on = 'customer_id')
X_train = X_train.merge(articles_df,how = 'left',on = 'article_id')

X_train.to_csv('Data/Preprocessed/X_train.csv',index=False)

No_customer = len(customer_df)

no_features = 3

### User : User simmilarities:
Customer_sim = np.empty((No_customer,No_customer, no_features))

for customer in range(No_customer):
    for customer_2 in range(No_customer):
        if customer == customer_2:
            continue
        else:
            Customer_sim[customer,customer_2,0] = customer_df.loc[customer]['age']-customer_df.loc[customer_2]['age']
            Customer_sim[customer,customer_2,1] = customer_df.loc[customer]['encoded_customer_status']-customer_df.loc[customer_2]['encoded_customer_status']
            Customer_sim[customer,customer_2,2] = customer_df.loc[customer]['encoded_news_frequency']-customer_df.loc[customer_2]['encoded_news_frequency']