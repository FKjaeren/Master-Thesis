
import torch
import pandas as pd
from Src.Create_recommendations import Get_Recommendations
import numpy as np
import pickle
from Src.deepFM import DeepFactorizationMachineModel
from Src.MLP_Model import MultiLayerPerceptronArchitecture
import yaml
from yaml.loader import SafeLoader
np.random.seed(42)
test_df_negatives = pd.read_csv('Data/Preprocessed/test_with_negative_subset_part2.csv', nrows = 120000000)
test_df_negatives = test_df_negatives[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name','graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status','fashion_news_frequency', 'age', 'postal_code','target']]
test_df = pd.read_csv('Data/Preprocessed/test_df_subset.csv')
#test_df = test_df_negatives[test_df_negatives['target']==1]
with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)
#model = DeepFactorizationMachineModel(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = 'cpu')
model = MultiLayerPerceptronArchitecture(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = 'cpu')
model.load_state_dict(torch.load('Models/MLP_model.pth'))
#model = torch.load('Models/DeepFM_modelV2.pth')
num_recommendations = 12
#chunksize = 200000
#test_full_data_path = '../../../../../../work3/s174478/Data/test_dataset_with_negative_part1.csv'
batch_size = hparams["batch_size"]
c = 328.0
c = 179700.
c = 299185.0
#887.0
#914.0
#1553.0
#1877.0
#348352.0
#348393.0
#348927.0
#349008.0
#350330.0
test_df_negatives = test_df_negatives[test_df_negatives['article_id']<number_uniques_dict['n_products']]
test_df_negatives = test_df_negatives[test_df_negatives['price']<number_uniques_dict['n_prices']]
test_df_negatives = test_df_negatives[test_df_negatives['prod_name']<number_uniques_dict['n_prod_names']]
test_df_negatives = test_df_negatives[test_df_negatives['department_name']<number_uniques_dict['n_departments']]
customers = test_df_negatives.customer_id.unique()

count_C = 0
one_accuracy_all = []
twelve_accuracy_all = []
for c in customers:
    #batch_size=test_df_negatives_temp.shape[0]
    temp_accuracy = []
    test_df_temp = test_df[test_df["customer_id"] == c]
    test_df_negatives_temp = test_df_negatives[test_df_negatives["customer_id"]==c]
    test_df_negatives_temp = test_df_negatives_temp.drop(columns = ['target'])
    test_df_negatives_temp = test_df_negatives_temp.sample(frac = 1)
    if(test_df_negatives_temp.shape[0] < 26000):
        continue
    elif(test_df_temp.empty):
        continue
    else:
        items, true_values = Get_Recommendations(customer_id = c, model = model, test_set=test_df_temp, test_full_set=test_df_negatives_temp, num_recommendations = num_recommendations)
        if any(x in items for x in true_values):
            accuracy = 1
        else:
            accuracy = 0
        one_accuracy_all.append(accuracy)
        temp_accuracy = sum(items==i for i in true_values).bool()
        if(num_recommendations <= len(true_values)):
            temp_accuracy_final = sum(temp_accuracy)/num_recommendations
        else:
            temp_accuracy_final = sum((np.sort(temp_accuracy)[::-1][0:len(true_values)]))/len(true_values)

        twelve_accuracy_all.append(temp_accuracy_final)
        count_C +=1


one_accuracy_all_final = sum(one_accuracy_all)/len(customers)
twelve_accuracy_all_final = sum(twelve_accuracy_all)/len(customers)
one_accuracy_all_final_subset = sum(one_accuracy_all)/count_C
twelve_accuracy_all_final_subset = sum(twelve_accuracy_all)/count_C

print("The accuracy at hitting one correct recommendation is in subset is: ",one_accuracy_all_final_subset*100, "%")
print("The accuracy at hitting 12 accurate recommendations is in subset is: ",twelve_accuracy_all_final_subset*100,"%")
print(f"Where {len(one_accuracy_all)} times N recommendations have been made")
print(f"Test: Where {len(twelve_accuracy_all)} times N recommendations have been made")
print(f"There are {count_C} customers with ~ 56000 recommendations being made")
print(f"For weight combos: FM: {hparams['fm_weight']} MLP: {hparams['mlp_weight']}")
