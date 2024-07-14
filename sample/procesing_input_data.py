import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def preprocess_input_data(data):
    # Organize the columns
    column_names = ['is_tv_subscriber_pred', 'is_movie_package_subscriber_pred', 'subscription_age_pred',
                    'bill_avg_pred', 'reamining_contract_pred', 'service_failure_count_pred',
                    'download_avg_pred', 'upload_avg_pred', 'download_over_limit_pred']

    # Convert to DataFrame for consistency
    df = pd.DataFrame([data], columns=column_names)

    # Rename the columns to those used when training the model
    df.columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
                  'bill_avg', 'reamining_contract', 'service_failure_count',
                  'download_avg', 'upload_avg', 'download_over_limit']

    # Convert bool values to int using the map function
    df['is_tv_subscriber'] = df['is_tv_subscriber'].astype(int)
    df['is_movie_package_subscriber'] = df['is_movie_package_subscriber'].astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # OneHotEncoder for categorical features if there are any
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_categorical_cols = encoder.fit_transform(df[categorical_cols])
        encoded_categorical_df = pd.DataFrame(encoded_categorical_cols,
                                              columns=encoder.get_feature_names_out(categorical_cols))
        encoded_categorical_df.reset_index(drop=True, inplace=True)
    else:
        encoded_categorical_df = pd.DataFrame()

    df.reset_index(drop=True, inplace=True)

    # Combine processed features
    processed_features = pd.concat([df[numeric_cols], encoded_categorical_df], axis=1)

    return processed_features