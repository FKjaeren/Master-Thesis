import streamlit as st
import pandas as pd
import torch
#from Create_recommendations import Get_Recommendations
from deepFM import DeepFactorizationMachineModel
from CreateRecommendationData import load_recommendation_data
import pickle
import os
from google.cloud import storage
import io

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


#@st.cache
def make_prediction(dcustomer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, Amount_input, customer_input, state_dict):

    model = DeepFactorizationMachineModel(field_dims = train_df.columns, embed_dim=26, n_unique_dict = number_uniques_dict, device = 'cpu', batch_size=1,dropout=0.2677)

    model.load_state_dict(state_dict)

    batch_size = 128
    _, full_data = load_recommendation_data(customer_input, None, customer_data, article_data, batch_size=batch_size, train_df=train_df)

    data_without_target = full_data[:,:20]
    outputs,_ = model(data_without_target)
    outputs = outputs.detach()
    conf, idx = torch.topk(outputs, Amount_input)

    non_encoders_ids = Article_Id_Encoder.inverse_transform(idx.reshape(-1,1))
    non_encoders_ids = non_encoders_ids.reshape(-1).astype(int)




    products = article_data_raw[article_data_raw['article_id'].isin(non_encoders_ids)]

    product_names = products['prod_name'].values
    product_colors = products['colour_group_name'].values

    return product_names, product_colors, conf


st.title("Make a customer recommendation")

customer_input = st.number_input(
    "Enter a customer id 👇 such as: 95084",
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

customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, state_dict = load_data(storage_client)

if (customer_input and Amount_input):
    data_load_state = st.text('Making Recommendation')
    product_names, product_colors, conf = make_prediction(customer_data, article_data, train_df, article_data_raw, number_uniques_dict, Article_Id_Encoder, Amount_input, customer_input,state_dict)

    st.text(f'The {str(Amount_input)} recommendations is product: {product_names} in color: {product_colors}')
    st.text(f'with confidence {str(conf.numpy())} %')