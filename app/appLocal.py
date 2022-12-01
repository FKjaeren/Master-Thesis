import streamlit as st
import pandas as pd
import torch
#from Create_recommendations import Get_Recommendations
from deepFM import DeepFactorizationMachineModel
from CreateRecommendationData import load_recommendation_data
import pickle
import os

data = pd.read_csv('../Data/Preprocessed/test_df_subset_final.csv')


st.title("Make a customer recommendation")
data_path = "Data/Preprocessed/"
def load_data(customer_id):
    data = pd.read_csv(data_path)
    data = data[data["customer_id"] == customer_id]
    return data

customer_input = st.number_input(
    "Enter a customer id 👇 such as: 95323",
    #label_visibility=st.session_state.visibility,
    #disabled=st.session_state.disabled,
    #placeholder=st.session_state.placeholder,
    )
Amount_input = st.number_input(
    "Enter amount of recommendations to be made 👇",
    #label_visibility=st.session_state.visibility,
    #disabled=st.session_state.disabled,
    #placeholder=st.session_state.placeholder,
    )

customer_input = int(customer_input)
Amount_input = int(Amount_input)

data_load_state = st.text('Making Recommendation')


customer_data = pd.read_csv('../Data/Preprocessed/customer_df_numeric_subset.csv')
article_data = pd.read_csv('../Data/Preprocessed/article_df_numeric_subset.csv')

train_df = pd.read_csv('../Data/Preprocessed/train_df_subset.csv')

batch_size = 128
data, full_data = load_recommendation_data(customer_input, data, customer_data, article_data, batch_size=batch_size, train_df=train_df)

with open(r"../Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

model = DeepFactorizationMachineModel(field_dims = data.columns, embed_dim=26, n_unique_dict = number_uniques_dict, device = 'cpu', batch_size=1,dropout=0.2677)
model.load_state_dict(torch.load('../Models/DeepFM_model_Final.pth'))
outputs,_ = model(full_data)
outputs = outputs.detach()
conf, idx = torch.topk(outputs, Amount_input)

with open(r"../Models/Article_Id_Encoder_subset.sav", "rb") as input_file:
    Article_Id_Encoder = pickle.load(input_file)

non_encoders_ids = Article_Id_Encoder.inverse_transform(idx.reshape(-1,1))
non_encoders_ids = non_encoders_ids.reshape(-1).astype(int)
article_data_raw = pd.read_csv('../Data/Raw/articles_subset.csv')

products = article_data_raw[article_data_raw['article_id'].isin(non_encoders_ids)]

product_names = products['prod_name'].values
product_colors = products['colour_group_name'].values

st.text(f'The {str(Amount_input)} recommendations is product: {product_names} in color: {product_colors}')
st.text(f'with confidence {str(conf.numpy())} %')