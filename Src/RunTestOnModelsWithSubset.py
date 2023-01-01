
import torch
import pandas as pd
from Src.Create_recommendations import Get_Recommendations
import numpy as np
import pickle
from Src.deepFM import DeepFactorizationMachineModel
from Src.MLP_Model import MultiLayerPerceptronArchitecture
from Src.FMModel import FactorizationMachineModel
import yaml
from yaml.loader import SafeLoader
np.random.seed(42)
#Due to local constraints only 120000000 rows of the data is loaded
test_df_negatives = pd.read_csv('Data/Preprocessed/test_with_negative_subset_part2.csv', nrows = 120000000)
test_df_negatives = test_df_negatives[['customer_id', 'article_id', 'price', 'sales_channel_id', 'day',
    'month', 'year', 'season', 'prod_name','graphical_appearance_name', 'colour_group_name', 'department_name',
    'club_member_status','fashion_news_frequency', 'age', 'postal_code','target']]
test_df = pd.read_csv('Data/Preprocessed/test_df_subset.csv')

with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)
#Initialize the correct model based on the config file
if(hparams['model'] == 'DeepFM'):
    model = DeepFactorizationMachineModel(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = hparams["device"])
elif(hparams['model'] == 'MLP'):
    model = MultiLayerPerceptronArchitecture(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = hparams["device"])
elif(hparams['model']=='FM'):
    model = FactorizationMachineModel(field_dims = test_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = hparams["device"])
else:
    print('Model type input is not recognized')

with open(r'Models/Prod_Name_Encoder_subset.sav', "rb") as input_file:
    Prod_Name_Encoder = pickle.load(input_file)

PATH = hparams["model_path"]
model.load_state_dict(torch.load(PATH))
num_recommendations = 12

batch_size = hparams["batch_size"]
number_params = 0
for parameter in model.parameters():
    number_params += len(parameter)
print(number_params)
#c = 328.0
#c = 179700.
#c = 299185.0
c=294142.0
#887.0
#914.0
#1553.0
#1877.0
#348352.0
#348393.0
#348927.0
#349008.0
#350330.0
#test_df_negatives = test_df_negatives[test_df_negatives['article_id']<number_uniques_dict['n_products']]
#test_df_negatives = test_df_negatives[test_df_negatives['price']<number_uniques_dict['n_prices']]
#test_df_negatives = test_df_negatives[test_df_negatives['prod_name']<number_uniques_dict['n_prod_names']]
#test_df_negatives = test_df_negatives[test_df_negatives['department_name']<number_uniques_dict['n_departments']]

## Calculate mAP(1) and mAP(12) for customers
customers = test_df_negatives.customer_id.unique()
correct_predicted_customer = []
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
    #Only caluclate for customers where every product is in subset
    if(test_df_negatives_temp.shape[0] < 56000):
        continue
    elif(test_df_temp.empty):
        continue
    else:
        items, true_values = Get_Recommendations(customer_id = c, model = model, test_set=test_df_temp, test_full_set=test_df_negatives_temp, num_recommendations = num_recommendations)
        if any(x in items.numpy() for x in true_values):
            accuracy = 1
            correct_predicted_customer.append(c)
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


one_accuracy_all_final_subset = sum(one_accuracy_all)/count_C
twelve_accuracy_all_final_subset = sum(twelve_accuracy_all)/count_C

with open(f'Results/correctlypredicted_customers_with_{hparams["model"]}.txt', 'w') as fp:
    for item in correct_predicted_customer:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Correctly predicted customers have been saved')

print("The accuracy at hitting one correct recommendation is in subset is: ",one_accuracy_all_final_subset*100, "%")
print("The accuracy at hitting 12 accurate recommendations is in subset is: ",twelve_accuracy_all_final_subset.item()*100,"%")
print(f"Where {len(one_accuracy_all)} times N recommendations have been made")
print(f"Test: Where {len(twelve_accuracy_all)} times N recommendations have been made")
print(f"There are {count_C} customers with ~ 56000 recommendations being made")
print(f"For weight combos: FM: {hparams['fm_weight']} MLP: {hparams['mlp_weight']}")
print(f"This test was performed with the model :{model}")



