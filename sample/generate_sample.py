import pandas as pd
import numpy as np

# Create a DataFrame with 100 rows of data
data = {
    'is_tv_subscriber_pred': np.random.choice([True, False], 100),
    'is_movie_package_subscriber_pred': np.random.choice([True, False], 100),
    'subscription_age_pred': np.random.uniform(0, 10, 100),
    'bill_avg_pred': np.random.randint(20, 150, 100),
    'reamining_contract_pred': np.random.uniform(0, 24, 100),
    'service_failure_count_pred': np.random.randint(0, 5, 100),
    'download_avg_pred': np.random.uniform(0, 500, 100),
    'upload_avg_pred': np.random.uniform(0, 100, 100),
    'download_over_limit_pred': np.random.randint(0, 10, 100),
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_path = 'batch_predictions.csv'
df.to_csv(csv_path, index=False)

print(f"CSV file saved to {csv_path}")