
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_path = 'Data/Raw/'

articles_df = pd.read_csv(data_path+'articles.csv')
customer_df = pd.read_csv(data_path+'customers.csv')
transactions_df = pd.read_csv(data_path+'transactions_train.csv')


df_a = articles_df[['article_id', 'prod_name', 'product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'section_name']]



transactions_df.t_dat = pd.to_datetime(transactions_df.t_dat)

sold_count = transactions_df['article_id'].value_counts()
sold_count=sold_count.reset_index()
sold_count.rename(columns = {"article_id":"sold_count","index":"article_id"}, inplace=True)

df_sold = df_a.merge(sold_count, on= 'article_id', how='left')
df_sold = df_sold.sort_values(by='sold_count', ascending=False)
# Maybe we need to drop articles not sold in the trainin period. But we could also keep to make more sale overall with recommendations.

# Lets see if there are any popular products in the collection
df_sold.dropna(inplace=True)
df_sold

top500 = df_sold.iloc[:500]

pivot = pd.pivot_table(top500, index= ["prod_name"], values='sold_count', aggfunc=np.sum)
pivot = pivot.reset_index()
top500.sort_values("prod_name", ascending=False)

ax = sns.barplot(x="prod_name", y="sold_count", data=top500.iloc[0:29])
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()
# We could display in % sold instead of total

