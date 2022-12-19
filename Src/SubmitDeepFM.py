import pandas as pd
import numpy as np
import torch
#from Create_recommendations import Get_Recommendations
from deepFM import DeepFactorizationMachineModel
from CreateRecommendationData import load_recommendation_data
import pickle
import yaml
from yaml.loader import SafeLoader

with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

data1 = pd.read_csv('Data/Preprocessed/train_df.csv')
data2 = pd.read_csv('Data/Preprocessed/valid_df.csv')
data3 = pd.read_csv('Data/Preprocessed/test_df.csv')

data = pd.concat([data1,data2,data3],axis = 0)
data = data.drop(columns=["FN", "Active","target","product_type_name","index_group_name"])

Amount_input = 12
Amount_input = int(Amount_input)


customer_data = pd.read_csv('Data/Preprocessed/customer_df_numeric.csv')
article_data = pd.read_csv('Data/Preprocessed/article_df_numeric.csv')

train_df = pd.read_csv('Data/Preprocessed/train_df_subset.csv')

batch_size = None
customers = customer_data.customer_id.unique()
customers_ids = []
predictions = []
results = {"customer_id":customers_ids,"prediction":predictions}

modelpath = hparams["model_path"]
modelpath = "Models/DeepFM_modelV2.pth"
model = torch.load(modelpath)
for c in customers:
    customer_input = c
    data, full_data = load_recommendation_data(customer_input, data, customer_data, article_data, batch_size=batch_size, train_df=train_df)

    with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
        number_uniques_dict = pickle.load(input_file)

    #model = DeepFactorizationMachineModel(field_dims = data.columns, embed_dim=26, n_unique_dict = number_uniques_dict, device = 'cpu', batch_size=1,dropout=0.2677)
    #model.load_state_dict(torch.load('Models/DeepFM_model_Final.pth'))
    outputs,_ = model(full_data)
    outputs = outputs.detach()
    conf, idx = torch.topk(outputs, Amount_input)
    customers_ids.append(c)
    predictions.append(idx)
#idx, conf = Get_Recommendations(customer_input, model=model, test_set=data, test_full_set = full_data, test_full_data_path=None, chunksize=None, batch_size = 1, num_recommendations = Amount_input, iter_data = False)
results = {"customer_id":customers_ids,"prediction":predictions}
final_df = pd.DataFrame(results)
final_df.to_csv('Results/KagglePredctions.csv',index=False)
