import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_path = 'Data/Preprocessed/'


df_a = pd.read_csv(data_path+'df_a.csv', index_col=0)
df_c = pd.read_csv(data_path+'df_c.csv', index_col=0)
df_t = pd.read_csv(data_path+'df_t.csv', index_col=0)


# Start exploration of data

# calculate number of sold products
sold_count = df_t['article_id'].value_counts()
sold_count=sold_count.reset_index()
sold_count.rename(columns = {"article_id":"sold_count","index":"article_id"}, inplace=True)

# create a dataframe with articles and sold count
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


# popular departments
pivot2 = pd.pivot_table(df_sold, index= ["department_name"], values='sold_count', aggfunc=np.sum)
pivot2 = pivot2.reset_index()
pivot2 = pivot2.sort_values("sold_count", ascending=False)

ax = sns.barplot(x="department_name", y="sold_count", data=pivot2.iloc[0:29])
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()



# Popular color?

pivot = pd.pivot_table(top500, index= ["colour_group_name"], values='sold_count', aggfunc=np.sum)
pivot = pivot.reset_index()
pivot = pivot.sort_values("sold_count", ascending=False)

ax = sns.barplot(x="colour_group_name", y="sold_count", data=pivot.iloc[0:29])
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()


# 4 plots with product type name in the seasons
prod_sea = df_t.merge(df_a, on= 'article_id', how='left')\
            .groupby(['season' , 'product_type_no' ,'product_type_name'])\
                .agg({'product_type_no': 'count'}).rename(columns={'product_type_no': 'qty'}).reset_index()


fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
fig.suptitle('Products sold in the seasons')

# Winter
Winter = prod_sea.loc[prod_sea['season'] == "Winter"].sort_values(by = 'qty', ascending = False).head(10)
sns.barplot(ax=axes[0, 0], x=Winter.product_type_name, y=Winter.qty)
axes[0,0].set_title("Winter")

# Spring
Spring = prod_sea.loc[prod_sea['season'] == "Spring"].sort_values(by = 'qty', ascending = False).head(10)
sns.barplot(ax=axes[0, 1], x=Spring.product_type_name, y=Spring.qty)
axes[0,1].set_title("Spring")

# Summer
Summer = prod_sea.loc[prod_sea['season'] == "Summer"].sort_values(by = 'qty', ascending = False).head(10)
sns.barplot(ax=axes[1, 0], x=Summer.product_type_name, y=Summer.qty)
axes[1,0].set_title("Summer")

# Autumn
Autumn = prod_sea.loc[prod_sea['season'] == "Autumn"].sort_values(by = 'qty', ascending = False).head(10)

sns.barplot(ax=axes[1, 1], x=Autumn.product_type_name, y=Autumn.qty)

axes[1,1].set_title("Autumn")
fig.set_xticklabels(rotation=65, horizontalalignment='right')
plt.show()

################ 
#Seasons plot 2



c_df = pd.merge(df_t, df_c.drop("postal_code", axis=1), on='customer_id', how='inner')
c_df['age_groups'] = pd.cut(c_df['age'], bins=[16, 20, 30, 50, 70,99], labels = ['teen' , 'young' , 'middle-aged' , 'senior', 'old'])
# 4 plots with product type name in the seasons
age_sea = c_df.groupby(['season' , 'age_groups'])\
                .agg({'age_groups': 'count'}).rename(columns={'age_groups': 'qty'}).reset_index()


fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
fig.suptitle('Number from age_groups who bought in the seasons')

# Winter
Winter = age_sea.loc[age_sea['season'] == "Winter"].sort_values(by = 'qty', ascending = False)
sns.barplot(ax=axes[0, 0], x=Winter.age_groups, y=Winter.qty)
axes[0,0].set_title("Winter")

# Spring
Spring = age_sea.loc[age_sea['season'] == "Spring"].sort_values(by = 'qty', ascending = False)
sns.barplot(ax=axes[0, 1], x=Spring.age_groups, y=Spring.qty)
axes[0,1].set_title("Spring")

# Summer
Summer = age_sea.loc[age_sea['season'] == "Summer"].sort_values(by = 'qty', ascending = False)
sns.barplot(ax=axes[1, 0], x=Summer.age_groups, y=Summer.qty)
axes[1,0].set_title("Summer")

# Autumn
Autumn = age_sea.loc[age_sea['season'] == "Autumn"].sort_values(by = 'qty', ascending = False)

sns.barplot(ax=axes[1, 1], x=Autumn.age_groups, y=Autumn.qty)

axes[1,1].set_title("Autumn")
fig.set_xticklabels(rotation=65, horizontalalignment='right')
plt.show()


# Er der nogen af aldersgrupperne der køber de samme ting?
# Hvilke aldersgrupper køber mest?
# Is membership a factor to buy special products or just more products. 
# What are the most popular products during the different season, maybe sequential network here?

# Lets look at customers
df_c.isnull().sum().sort_values(ascending=False)
# there are many missing values in the memberships
# how is the distriobution of customers in age
sns.histplot(data=df_c, x="age",  binwidth = 1)
set(xlim=(0,100))
plt.show()

# We need to make age groups instead of age alone to make usefull analysis


c_df = pd.merge(df_t, df_c.drop("postal_code", axis=1), on='customer_id', how='inner')
c_df['age_groups'] = pd.cut(c_df['age'], bins=[16, 20, 30, 50, 70,99], labels = ['teen' , 'young' , 'middle-aged' , 'senior', 'old'])

# We want as many features describing how the customer relationships are. Both from the differnet customers and the differnet articles.
#dep, price, artice, cus id colour age features
sns.histplot(data=df_t, x="price",  binwidth = 1)
set(xlim=(0,100))
plt.show()




