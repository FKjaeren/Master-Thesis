import pandas as pd
df = pd.read_csv('Data/Raw/transactions_train_subset.csv')
splitrange = round(0.8*len(df['customer_id']))
splitrange2 = round(0.975*len(df['customer_id']))


train_with_negative = pd.read_csv('Data/Preprocessed/train_df_subset.csv')
test_df_encoded = pd.read_csv('Data/Preprocessed/test_df_subset.csv')

test = df.iloc[splitrange2+1:]
train_customers = df.iloc[:splitrange].customer_id.unique()
train_articles_with_negative = train_with_negative.article_id.unique()
article_train = df.iloc[:splitrange].article_id.unique()

test_sub = test[['customer_id','article_id']]
customers = test_sub.customer_id.unique()[0:12000]
customers_encoded = test_df_encoded.customer_id.unique()[0:12000]

test_df_sub = test_df_encoded[test_df_encoded["customer_id"].isin(customers_encoded)]


final_idx = test_sub.index[test_sub['customer_id'].isin(customers)].tolist()
final_idx_encoded = test_df_encoded.index[test_df_encoded['customer_id'].isin(customers_encoded)].tolist()

articles = test_sub.article_id.loc[final_idx].unique()
articles_encoded = test_df_encoded.article_id[final_idx_encoded].unique()

c = sum(customer in customers for customer in train_customers)
a = sum(article in articles for article in article_train)
a_neg = sum(article in articles_encoded for article in train_articles_with_negative)

print(f"Number of customers in the test set that were also in the trainingset is: {c}. Therefore there are {12000-c}, cold start cases in regards to customers.")
print(f"Number of articles in the test set that were also in the trainingset is: {a}. Therefore there are {len(articles)-a}, cold start cases in regards to customers.")
print(f"Number of articles in the test set that were also in the trainingset included the negative samples of random articles is: {a_neg}. Therefore there are {len(articles)-a_neg}, cold start cases in regards to customers.")

test_df_sub = test_df_encoded[test_df_encoded["customer_id"].isin(customers_encoded)]

customer_transaction_analysis = test_df_sub[['customer_id','article_id']].groupby('customer_id').count()
avg_transactions_made = customer_transaction_analysis.mean()
max_transactions_made = customer_transaction_analysis.max()
min_transactions_made = customer_transaction_analysis.min()
median_transactions_made = customer_transaction_analysis.median()

