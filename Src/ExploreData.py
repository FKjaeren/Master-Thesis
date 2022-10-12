import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

products = pd.read_csv('Data/Raw/articles.csv')
train_df = pd.read_csv('Data/Preprocessed/train_df.csv')

products.columns

products.department_name.hist()

sns.histplot(data = products, x = 'product_type_name' )
# show the graph
plt.xticks(rotation = 0.45)
plt.show()  