import pandas as pd
from CreateNegativeSamples import CreateNegativeSamples


def load_recommendation_data(customer_input, data, customer_data, article_data, batch_size, train_df):
    data_subset_with_negative = CreateNegativeSamples(None, train_df, num_negative_samples=article_data.article_id.nunique(), type_df = 'Test', method = 'OneCustomerNegSamples', customer_id = customer_input, article_df=article_data, customer_df = customer_data, batch_size = batch_size)
    return None, data_subset_with_negative
    