import torch
import pandas as pd
from Create_recommendations import Get_Recommendations
import numpy as np
import pickle
from deepFM import DeepFactorizationMachineModel
import yaml
from yaml.loader import SafeLoader

test_df_negatives = pd.read_csv('../../../../../../work3/s174478/Data/test_dataset_with_negative_part1.csv', nrows = 60000000)
test_df_negatives = test_df_negatives[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name','graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status','fashion_news_frequency', 'age', 'postal_code','target']]
#test_df_negatives = pd.read_csv('Data/Preprocessed/test_with_negative_subset_part1.csv', nrows = 60000000)
test_df = pd.read_csv('../../../../../../work3/s174478/Data/test_dataset_subset.csv')
#test_df = test_df_negatives[test_df_negatives['target']==1]
customers = test_df.customer_id.unique()

with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

one_accuracy_all = []
twelve_accuracy_all = []

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)
#model = DeepFactorizationMachineModel(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = 'cpu')
#model.load_state_dict(torch.load('Models/DeepFM_model_Final.pth'))
model = torch.load('Models/DeepFM_modelV2.pth')
num_recommendations = 12
chunksize = 200000
test_full_data_path = '../../../../../../work3/s174478/Data/test_dataset_with_negative_part1.csv'
batch_size=1
for c in customers:
    temp_accuracy = []
    test_df_temp = test_df[test_df["customer_id"] == c]
    test_df_negatives_temp = test_df_negatives[test_df_negatives["customer_id"]==c]
    if(test_df_negatives_temp.shape[0] == 1):
        continue
    else:
        items, true_values = Get_Recommendations(customer_id = c, model = model, test_set=test_df_temp, test_full_set=test_df_negatives_temp, test_full_data_path =None ,chunksize = None,batch_size=batch_size, num_recommendations = num_recommendations, iter_data = False)
        if any(x in items for x in true_values):
            accuracy = 1
        else:
            accuracy = 0
        one_accuracy_all.append(accuracy)
        for i in items:
            if i in true_values:
                temp_accuracy.append(1)
            else:
                temp_accuracy.append(0)
        if(num_recommendations <= len(true_values)):
            temp_accuracy = sum(temp_accuracy)/num_recommendations
        else:
            temp_accuracy = sum((np.sort(temp_accuracy)[0:len(true_values)]))/len(true_values)

        twelve_accuracy_all.append(temp_accuracy)

one_accuracy_all = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all = sum(twelve_accuracy_all)/len(customers)

print("The accuracy at hitting one correct recommendation is: ",one_accuracy_all, "%")
print("The accuracy at hitting 12 accurate recommendations is ",twelve_accuracy_all,"%")

