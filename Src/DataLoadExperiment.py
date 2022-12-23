import streamlit as st
import pandas as pd
import numpy as np
import torch
#from Create_recommendations import Get_Recommendations
from deepFM import DeepFactorizationMachineModel
from CreateRecommendationData import load_recommendation_data
import pickle
import os
from google.cloud import storage
import io


import yaml
from yaml.loader import SafeLoader
data = pd.read_csv('Data/Preprocessed/test_df_subset.csv')
with open('config/experiment/exp1.yaml') as f:
    hparams = yaml.load(f, Loader=SafeLoader)

import time
startTimeLocal = time.time()

customer_data = pd.read_csv('Data/Preprocessed/customer_df_numeric_subset.csv')
article_data = pd.read_csv('Data/Preprocessed/article_df_numeric_subset.csv')
article_data_raw = pd.read_csv('Data/Raw/articles_subset.csv')

train_df = pd.read_csv('Data/Preprocessed/train_df_subset.csv')
with open(r'Models/Article_Id_Encoder_subset.sav', "rb") as input_file:
    Article_Id_Encoder = pickle.load(input_file)

executionTimeLocal = (time.time() - startTimeLocal)
print('Local execution time in seconds: ' + str(executionTimeLocal))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "masterthesis-366109-0c44ff859b6e.json"
PROJECT = "MasterThesis"
REGION = "eu-west1"

storage_client = storage.Client()

#@st.cache
def load_data(storage_client):
    DeepFM_model_blob_name = "DeepFM_model_Final.pth"
    Article_Id_Encoder_blob_name = "Article_Id_Encoder_subset.sav"
    #test_df_subset_blob_name = "test_df_subset_final.csv"
    number_uniques_dict_blob_name = "number_uniques_dict_subset.pickle"
    customer_df_numeric_blob_name = "customer_df_numeric_subset.csv"
    article_df_numeric_blob_name = "article_df_numeric_subset.csv"
    articles_raw_blob_name = "articles_subset.csv"
    train_df_subset_blob_name = "train_df_subset.csv"

    Data_bucket_name = "preprocessed_rec_data"
    Model_bucket_name = "model_container_masters"
    Raw_Data_bucket_name = "raw_rec_data"

    Data_bucket = storage_client.bucket(Data_bucket_name)
    Model_bucket = storage_client.bucket(Model_bucket_name)
    Raw_Data_bucket = storage_client.bucket(Raw_Data_bucket_name)

    DeepFM_model_blob = Model_bucket.blob(DeepFM_model_blob_name)
    Article_Id_Encoder_blob = Model_bucket.blob(Article_Id_Encoder_blob_name)
    #test_df_subset_blob = Data_bucket.blob(test_df_subset_blob_name)
    number_uniques_dict_blob = Data_bucket.blob(number_uniques_dict_blob_name)
    customer_df_numeric_blob = Data_bucket.blob(customer_df_numeric_blob_name)
    article_df_numeric_blob = Data_bucket.blob(article_df_numeric_blob_name)
    articles_raw_blob = Raw_Data_bucket.blob(articles_raw_blob_name)
    train_df_subset_blob = Data_bucket.blob(train_df_subset_blob_name)

    #with test_df_subset_blob.open("r") as f:
    #    data = pd.read_csv(f)
    
    with customer_df_numeric_blob.open("r") as f:
        customer_data = pd.read_csv(f)
    with article_df_numeric_blob.open("r") as f:
        article_data = pd.read_csv(f)

    with train_df_subset_blob.open("r") as f:
        train_df = pd.read_csv(f)

    with articles_raw_blob.open("r") as f:
        article_data_raw = pd.read_csv(f)

    pickle_in = number_uniques_dict_blob.download_as_string()
    number_uniques_dict = pickle.loads(pickle_in)

    pickle_in = Article_Id_Encoder_blob.download_as_string()
    buffer = io.BytesIO(pickle_in)
    Article_Id_Encoder = pickle.load(buffer)

    en_model = DeepFM_model_blob.download_as_string()
    buffer = io.BytesIO(en_model)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    return customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, state_dict

startTimeCloud = time.time()
customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, state_dict = load_data(storage_client)

executionTimeCloud = (time.time() - startTimeCloud)
print('Local execution time in seconds: ' + str(executionTimeCloud))