
import numpy as np
import pandas as pd
import tensorflow as tf
#from Src.BaselineFactorizationModel import SimpleRecommender

## Define dataset
def get_dataset(df):
    dummy_customer_tensor = tf.constant(df[['customer_id']].values, dtype =tf.string)
    article_tensor = tf.constant(df[['article_id']].values,dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((dummy_customer_tensor,article_tensor))
    
    dataset = dataset.batch(1)
    return dataset 



df = pd.read_csv('Data/Raw/transactions_train_subset.csv')
splitrange = round(0.8*len(df['customer_id']))
splitrange2 = round(0.975*len(df['customer_id']))




test = df.iloc[splitrange2+1:]

test_sub = test[['customer_id','article_id']]

customers = test_sub.customer_id.unique()[0:12000]
test_sub = test_sub[test_sub["customer_id"].isin(customers)]


customers = test_sub["customer_id"].unique()

# Load model

baseline_model = tf.keras.models.load_model('Models/test_baseline_model_with_tf_function2')
dataset = get_dataset(test_sub)

k= 12



#Loop for calculating mAP(1) and mAP(12)

one_accuracy_all = []
twelve_accuracy_all = []

for c in customers:

    temp_accuracy = []
    test_df_temp = test_sub[test_sub["customer_id"] == c]
    true_values = test_df_temp.article_id.unique()

    recommendations, scores = baseline_model.Customer_recommendation(tf.constant(np.array([[c]]), dtype=tf.string), tf.constant(k, dtype=tf.int32))
    if any(x in recommendations for x in true_values):
        accuracy = 1.0
    else:
        accuracy = 0.0
    one_accuracy_all.append(accuracy)
    for i in recommendations:
        if i in true_values:
            temp_accuracy.append(1)
        else:
            temp_accuracy.append(0)
    if(k <= len(true_values)):
        temp_accuracy = sum(temp_accuracy)/k
    else:
        temp_accuracy = sum((np.sort(temp_accuracy)[::-1][0:len(true_values)]))/len(true_values)

    twelve_accuracy_all.append(temp_accuracy)

one_accuracy_all = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all = sum(twelve_accuracy_all)/len(customers)





print("The accuracy at hitting one correct recommendation is: ",one_accuracy_all*100, "%")
print("The accuracy at hitting 12 accurate recommendations is ",twelve_accuracy_all*100,"%")