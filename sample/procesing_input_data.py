import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, Normalizer


def preprocess_input_data(data):
    # Organize column names
    column_names = ['is_tv_subscriber_pred', 'is_movie_package_subscriber_pred', 'subscription_age_pred',
                    'bill_avg_pred', 'reamining_contract_pred', 'service_failure_count_pred',
                    'download_avg_pred', 'upload_avg_pred', 'download_over_limit_pred']

    # Convert to DataFrame for consistency
    df = pd.DataFrame([data], columns=column_names)

    # Rename columns to match those used during model training
    df.columns = ['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
                  'bill_avg', 'reamining_contract', 'service_failure_count',
                  'download_avg', 'upload_avg', 'download_over_limit']

    # Convert bool values to int
    df['is_tv_subscriber'] = df['is_tv_subscriber'].astype(int)
    df['is_movie_package_subscriber'] = df['is_movie_package_subscriber'].astype(int)

    # Convert DataFrame to numpy array for Normalizer
    X = df.to_numpy()

    # Scale numeric columns using Normalizer
    normalizer = Normalizer(norm='l2')
    X_normalized = normalizer.transform(X)

    # Convert back to DataFrame with original column names
    df_normalized = pd.DataFrame(X_normalized, columns=df.columns)

    # OneHotEncoder for categorical features if there are any
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    if not categorical_cols.empty:
        encoder = OneHotEncoder(drop='first')
        encoded_categorical_cols = encoder.fit_transform(df[categorical_cols])
        encoded_categorical_df = pd.DataFrame(encoded_categorical_cols.toarray(),
                                              columns=encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_categorical_df = pd.DataFrame()

    # Combine processed features
    processed_features = pd.concat([df_normalized, encoded_categorical_df], axis=1)

    return processed_features
