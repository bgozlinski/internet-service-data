import pandas as pd
import numpy as np


def generate_data_with_two_decimal_points(num_rows=100, file_path='batch_predictions.csv'):
    """
    Generate a DataFrame with specified number of rows and save it to a CSV file with float values rounded to 2 decimal points.

    Args:
    num_rows (int): Number of rows of data to generate. Default is 100.
    file_path (str): Path to save the generated CSV file. Default is 'batch_predictions.csv'.

    Returns:
    str: Path to the saved CSV file.
    """
    # Create a DataFrame with specified number of rows of data
    data = {
        'is_tv_subscriber_pred': np.random.choice([True, False], num_rows),
        'is_movie_package_subscriber_pred': np.random.choice([True, False], num_rows),
        'subscription_age_pred': np.random.uniform(0, 10, num_rows),
        'bill_avg_pred': np.random.randint(20, 150, num_rows),
        'reamining_contract_pred': np.random.uniform(0, 24, num_rows),
        'service_failure_count_pred': np.random.randint(0, 5, num_rows),
        'download_avg_pred': np.random.uniform(0, 500, num_rows),
        'upload_avg_pred': np.random.uniform(0, 100, num_rows),
        'download_over_limit_pred': np.random.randint(0, 10, num_rows),
    }

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data)

    # Round float columns to 2 decimal places
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(3)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

    return file_path


# Generate the data and save to a CSV file
csv_file_path = generate_data_with_two_decimal_points(file_path='batch_predictions.csv')
print(f"CSV file saved to {csv_file_path}")
