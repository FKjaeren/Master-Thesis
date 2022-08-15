
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')


df_a = articles_df[['article_id', 'prod_name', 'product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'section_name']]


#datetime and create a month column
transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)
transactions_df['month'] =  pd.DatetimeIndex(transactions_df['t_dat']).month

transactions_df.loc[(transactions_df['month']>= 1) & (transactions_df['month'] <=2), 'season'] = 'Winter'
transactions_df.loc[(transactions_df['month'] == 12), 'season'] = 'Winter' 
transactions_df.loc[(transactions_df['month'] >= 3) & (transactions_df['month'] <=5), 'season'] = 'Spring' 
transactions_df.loc[(transactions_df['month'] >= 6) & (transactions_df['month'] <=8),'season'] = 'Summer' 
transactions_df.loc[(transactions_df['month'] >= 9) & (transactions_df['month'] <=11), 'season'] = 'Autumn' 


# calculate number of sold products
sold_count = transactions_df['article_id'].value_counts()
sold_count=sold_count.reset_index()
sold_count.rename(columns = {"article_id":"sold_count","index":"article_id"}, inplace=True)

# create a column with articles and sold count
df_sold = df_a.merge(sold_count, on= 'article_id', how='left')
df_sold = df_sold.sort_values(by='sold_count', ascending=False)
# Maybe we need to drop articles not sold in the trainin period. But we could also keep to make more sale overall with recommendations.
#df_sold.dropna(inplace=True)
#df_sold

# Lets see if there are any popular products in the collection


top500 = df_sold.iloc[:500]

pivot = pd.pivot_table(top500, index= ["prod_name"], values='sold_count', aggfunc=np.sum)
pivot = pivot.reset_index()
pivot = pivot.sort_values("sold_count", ascending=False)

ax = sns.barplot(x="prod_name", y="sold_count", data=pivot.iloc[0:29])
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()
# We could display in % sold instead of total

# Popular color?

pivot = pd.pivot_table(top500, index= ["colour_group_name"], values='sold_count', aggfunc=np.sum)
pivot = pivot.reset_index()
pivot = pivot.sort_values("sold_count", ascending=False)

ax = sns.barplot(x="colour_group_name", y="sold_count", data=pivot.iloc[0:29])
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()


# Er der nogen af aldersgrupperne der køber de samme ting?
# Hvilke aldersgrupper køber mest?
# Is membership a factor to buy special products or just more products. 
# What are the most popular products during the different season, maybe sequential network here?

# Lets look at customers
customer_df.isnull().sum().sort_values(ascending=False)
# there are many missing values in the memberships
# how is the distriobution of customers in age
sns.histplot(data=customer_df, x="age",  binwidth = 1)
set(xlim=(0,100))
plt.show()

# We need to make age groups instead of age alone to make usefull analysis


c_df = pd.merge(transactions_df, customer_df.drop("postal_code", axis=1), on='customer_id', how='inner')
labels = ['teen' , 'young' , 'middle-aged' , 'senior', 'old']
#['16-20', '20-30','30-50','50-70', '70+']
c_df['age_groups'] = pd.cut(c_df['age'], bins=[16, 20, 30, 50, 70,99], labels=labels)

