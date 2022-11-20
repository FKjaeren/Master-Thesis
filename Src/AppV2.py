import streamlit as st
import pandas as pd
import numpy as np
import torch
from Src.Create_recommendations import Get_Recommendations
from Src.deepFM import DeepFactorizationMachineModel
import pickle

st.title("Make a customer recommendation")
data_path = "Data/Preprocessed/"
def load_data(customer_id):
    data = pd.read_csv(data_path)
    data = data[data["customer_id"] == customer_id]
    return data

customer_input = st.number_input(
    "Enter a customer id 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
    )
Amount_input = st.number_input(
    "Enter amount of recommendations to be made 👇",
    label_visibility=st.session_state.visibility,
    disabled=st.session_state.disabled,
    placeholder=st.session_state.placeholder,
    )

data_load_state = st.text('Making Recommendation')
data, full_data = load_data(customer_input)

with open(r"Data/Preprocessed/number_uniques_dict_subset.pickle", "rb") as input_file:
    number_uniques_dict = pickle.load(input_file)

model = DeepFactorizationMachineModel(field_dims = data.columns, embed_dim=26, n_unique_dict = number_uniques_dict, device = 'cpu', batch_size=1,dropout=0.2677)
outputs = model(data)
idx, conf = Get_Recommendations(customer_input, model=model, test_set=data, test_full_set = full_data, batch_size = 1, num_recommendations = Amount_input):

data_load_state.text('The '+str(Amount_input)+'recommendations is: '+str(idx))