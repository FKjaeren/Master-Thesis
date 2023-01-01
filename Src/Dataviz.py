
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_path = 'Data/Raw/'

""" 
df_a = pd.read_csv(data_path+'article_df_numeric_subset.csv', index_col=0)
df_c = pd.read_csv(data_path+'df_c.csv', index_col=0)
df_t = pd.read_csv(data_path+'df_t.csv', index_col=0)
 """

df_a = pd.read_csv(data_path+'articles_subset.csv', index_col=0)
df_t = pd.read_csv(data_path + 'transactions_train.csv', index_col=0)



# calculate number of sold products
sold_count = df_t['article_id'].value_counts()
sold_count=sold_count.reset_index()
sold_count.rename(columns = {"article_id":"sold_count","index":"article_id"}, inplace=True)

# create a dataframe with articles and sold count
df_sold = df_a.merge(sold_count, on= 'article_id', how='left')
df_sold = df_sold.sort_values(by='sold_count', ascending=False)

# Lets see if there are any popular products in the collection


top20 = df_sold.iloc[:20]



pivot1 = pd.pivot_table(top20, index= ["prod_name"], values='sold_count', aggfunc=np.sum)
pivot1 = pivot1.reset_index()
pivot1 = pivot1.sort_values("sold_count", ascending=False)


ax = sns.barplot(x="sold_count", y="prod_name", data=pivot1)
#plt.setp(ax.get_xticklabels(), rotation=90)
ax.set_title("Top 15 most sold articles", fontsize=16)
ax.tick_params(axis="both", which = "major", labelsize=14)
""" 
for var in ax.containers:
    ax.bar_label(var, fontsize=9) """
plt.xlabel("Number of sold articles", fontsize= 16)

plt.ylabel("")
plt.show()


pivot2 = pd.pivot_table(top20, index= ["department_name"], values='sold_count', aggfunc=np.sum)
pivot2 = pivot2.reset_index()
pivot2 = pivot2.sort_values("sold_count", ascending=False)


ax = sns.barplot(x="sold_count", y="department_name", data=pivot2.iloc[:10])
#plt.setp(ax.get_xticklabels(), rotation=90)
ax.set_title("Top 10 most popular departments", fontsize =16)
ax.tick_params(axis="both", which = "major", labelsize=14)
""" 
for var in ax.containers:
    ax.bar_label(var, fontsize=9) """
plt.xlabel("Number of sold articles", fontsize = 16)
plt.ylabel("")
plt.show()

pivot3 = pd.pivot_table(top20, index= ["colour_group_name"], values='sold_count', aggfunc=np.sum)
pivot3 = pivot3.reset_index()
pivot3 = pivot3.sort_values("sold_count", ascending=False)


ax = sns.barplot(x="sold_count", y="colour_group_name", data=pivot3)
#plt.setp(ax.get_xticklabels(), rotation=90)
ax.set_title("Top 5 most popular colours", fontsize= 16)
ax.tick_params(axis="both", which = "major", labelsize=14)
""" 
for var in ax.containers:
    ax.bar_label(var, fontsize=9) """
plt.xlabel("Number of sold articles", fontsize = 16)
plt.ylabel("")
plt.show()

