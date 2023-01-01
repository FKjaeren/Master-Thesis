import streamlit as st
import pandas as pd
import numpy as np
import torch
#from Create_recommendations import Get_Recommendations
from FMModel import FactorizationMachineModel
from CreateRecommendationData import load_recommendation_data
import pickle

import yaml
from yaml.loader import SafeLoader
data = pd.read_csv('Data/Preprocessed/test_df_subset.csv')
with open('config/experiment/train_FM.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

st.title("Make a customer recommendation")
data_path = "Data/Preprocessed/"
def load_data(customer_id):
    data = pd.read_csv(data_path)
    data = data[data["customer_id"] == customer_id]
    return data

# Get customer input written in the app.
customer_input = st.number_input(
    "Enter a customer id 👇 such as: 299185",
    #label_visibility=st.session_state.visibility,
    #disabled=st.session_state.disabled,
    #placeholder=st.session_state.placeholder,
    )
# Get amount of recommendations to be made written in the app
Amount_input = st.number_input(
    "Enter amount of recommendations to be made 👇",
    #label_visibility=st.session_state.visibility,
    #disabled=st.session_state.disabled,
    #placeholder=st.session_state.placeholder,
    )

customer_input = int(customer_input)
Amount_input = int(Amount_input)

data_load_state = st.text('Making Recommendation')

# Load data
customer_data = pd.read_csv('Data/Preprocessed/customer_df_numeric_subset.csv')
article_data = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv')
article_data_raw = pd.read_csv('Data/Raw/articles_subset.csv')

train_df = pd.read_csv('Data/Preprocessed/train_df_subset.csv')
with open(r'Models/Article_Id_Encoder_subset.sav', "rb") as input_file:
    Article_Id_Encoder = pickle.load(input_file)

batch_size = 128
#data, full_data = load_recommendation_data(customer_input, data, customer_data, article_data, batch_size=batch_size, train_df=train_df)

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

def make_prediction(customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, Amount_input, customer_input, data):
    device = "cpu"
    model = FactorizationMachineModel(field_dims = train_df.columns, hparams=hparams, n_unique_dict = number_uniques_dict, device = device)
    model.load_state_dict(torch.load('Models/FM_modelV1.pth'))

    batch_size = hparams["batch_size"]
    _, full_data = load_recommendation_data(customer_input, data, customer_data, article_data, batch_size=batch_size, train_df=train_df)

    data_without_target = full_data[:,:16]
    outputs,_ = model(data_without_target)
    outputs = outputs.detach()
    conf, idx = torch.topk(outputs, Amount_input)

    #Decode the article ids
    non_encoders_ids = Article_Id_Encoder.inverse_transform(idx.reshape(-1,1))
    non_encoders_ids = non_encoders_ids.reshape(-1).astype(int)



    # Find product names from the articles look table
    products = article_data_raw[article_data_raw['article_id'].isin(non_encoders_ids)]

    product_names = products['prod_name'].values
    product_colors = products['colour_group_name'].values

    return product_names, product_colors, conf
# Wait to make predictions until a customer id and amount of recommedations have been given from the user of the app
if (customer_input and Amount_input):
    data_load_state = st.text('Making Recommendation')
    product_names, product_colors, conf = make_prediction(customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, Amount_input, customer_input, data)

    st.text(f'The {str(Amount_input)} recommendations is product: {product_names} in color: {product_colors}')
