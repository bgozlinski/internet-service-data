import os
from flask import Blueprint, jsonify, request
from src.database.models import Prediction
from src.database.db import get_db
import numpy as np
import joblib
import pandas as pd

# Create a Blueprint for the routes
route_bp = Blueprint('route', __name__)

# Define the absolute path to the models directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_models')

# Load the models using absolute paths
random_forest_model_path = os.path.join(model_dir, 'random_forest_model.pkl')
svm_model_path = os.path.join(model_dir, 'svm_model.pkl')

random_forest_model = joblib.load(random_forest_model_path)
svm_model = joblib.load(svm_model_path)


@route_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict the probability of churn for a customer using either the Random Forest or SVM model.

    The model to use is specified in the input JSON. The features for the prediction are extracted
    from the input JSON and used to make a prediction. The result, along with the input data, is
    saved in the database and returned as a JSON response.

    Returns:
        (json): A JSON response containing the model used and the prediction probability.
    """
    db = next(get_db())  # Get a database session
    data = request.get_json() # Get the input JSON data

    # Determine which model to use
    model_choice = data.get('model_choice', 'random_forest')

    # List of feature keys
    features = [
        'is_tv_subscriber_pred',
        'is_movie_package_subscriber_pred',
        'subscription_age_pred',
        'bill_avg_pred',
        'reamining_contract_pred',
        'service_failure_count_pred',
        'download_avg_pred',
        'upload_avg_pred',
        'download_over_limit_pred',
    ]

    # Extract feature values in a loop
    features = [data[key] for key in features]

    features = np.array([features])

    prediction_prob = None
    model_used = model_choice

    # Make the prediction using the selected model
    if model_choice == 'random_forest':
        # Predict using the Random Forest model
        prediction_prob = random_forest_model.predict_proba(features)[0, 1]
    elif model_choice == 'svm':
        # Predict using the SVM model
        svm_prob = svm_model.decision_function(features)
        prediction_prob = 1 / (1 + np.exp(-svm_prob))
        prediction_prob = prediction_prob[0]
    else:
        return jsonify({"error": "Invalid model choice"}), 400

    response = {
        'model_used': model_used,
        'prediction_prob': float(prediction_prob)
    }

    # Prepare the data for saving to the database
    prediction_data = data.copy()
    prediction_data.update({
        'prediction_prob': prediction_prob,
        'model_used': model_used
    })

    # Create a new Prediction object
    new_prediction = Prediction.from_dict(prediction_data)

    # Save the prediction to the database
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    return jsonify(response), 200


@route_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict the probability of churn for multiple customers using either the Random Forest or SVM model.

    The model to use is specified in the input CSV. The features for the predictions are extracted
    from the CSV and used to make predictions. The results, along with the input data, are saved
    in the database and returned as a JSON response.

    Returns:
        (json): A JSON response containing the model used and the prediction probabilities for each row in the CSV.
    """
    db = next(get_db())  # Get a database session

    # Check if the request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Ensure the file is a CSV
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File is not a CSV"}), 400

    try:
        data = pd.read_csv(file)

        # Ensure 'model_choice' column exists
        if 'model_choice' not in data.columns:
            data['model_choice'] = 'random_forest'  # Default to 'random_forest' if not specified

        responses = []

        for index, row in data.iterrows():
            model_choice = row['model_choice']
            features = np.array([
                row['is_tv_subscriber_pred'],
                row['is_movie_package_subscriber_pred'],
                row['subscription_age_pred'],
                row['bill_avg_pred'],
                row['reamining_contract_pred'],
                row['service_failure_count_pred'],
                row['download_avg_pred'],
                row['upload_avg_pred'],
                row['download_over_limit_pred']
            ]).reshape(1, -1)

            prediction_prob = None
            model_used = model_choice

            if model_choice == 'random_forest':
                prediction_prob = random_forest_model.predict_proba(features)[0, 1]
            elif model_choice == 'svm':
                svm_prob = svm_model.decision_function(features)
                prediction_prob = 1 / (1 + np.exp(-svm_prob))  # Applying logistic function to get probabilities
                prediction_prob = prediction_prob[0]
            else:
                return jsonify({"error": "Invalid model choice"}), 400

            response = {
                'model_used': model_used,
                'prediction_prob': float(prediction_prob)
            }

            new_prediction_data = row.to_dict()
            new_prediction_data.update({
                'prediction_prob': prediction_prob,
                'model_used': model_used
            })
            new_prediction = Prediction.from_dict(new_prediction_data)

            db.add(new_prediction)
            db.commit()
            db.refresh(new_prediction)

            responses.append(response)

        return jsonify(responses), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500