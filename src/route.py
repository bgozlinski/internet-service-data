import os
from flask import Blueprint, jsonify, request, render_template
from sample.procesing_input_data import preprocess_input_data
from src.database.models import Prediction
from src.database.db import get_db
import numpy as np
import joblib
import pandas as pd
import tensorflow as tf

# Create a Blueprint for the routes
route_bp = Blueprint('route', __name__)

# Define the absolute path to the models directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml_models')

# Define paths for all models
model_paths = {
    'gradient_boosting': os.path.join(model_dir, 'gradient_boosting_model.pkl'),
    'tensorflow_keras': os.path.join(model_dir, 'keras_model.h5'),
    'knn': os.path.join(model_dir, 'knn_model.pkl'),
    'mlp': os.path.join(model_dir, 'mlp_model.pkl'),
    'random_forest': os.path.join(model_dir, 'random_forest_model.pkl'),
    'svc': os.path.join(model_dir, 'svc_model.pkl'),
    'xgboost': os.path.join(model_dir, 'xgboost_model.pkl'),
    'logistic_regression': os.path.join(model_dir, 'logistic_regression.pkl'),
}

# Load all models
models = {
    'gradient_boosting': joblib.load(model_paths['gradient_boosting']),
    'tensorflow_keras': tf.keras.models.load_model(model_paths['tensorflow_keras']),
    'knn': joblib.load(model_paths['knn']),
    'mlp': joblib.load(model_paths['mlp']),
    'random_forest': joblib.load(model_paths['random_forest']),
    'svc': joblib.load(model_paths['svm']),
    'xgboost': joblib.load(model_paths['xgboost']),
    'logistic_regression': joblib.load(model_paths['logistic_regression']),
}


@route_bp.route('/')
def index():
    return render_template('index.html')


@route_bp.route('/single_input')
def single_input():
    return render_template('single_input.html')


@route_bp.route('/batch_input')
def batch_input():
    return render_template('batch_input.html')


@route_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict the probability of churn for a customer using either the Random Forest or SVM model.

    The model to use is specified in the input JSON or form data. The features for the prediction are extracted
    from the input and used to make a prediction. The result, along with the input data, is
    saved in the database and returned as a JSON response or rendered HTML template.

    Returns:
        (json): A JSON response containing the model used and the prediction probability, or rendered HTML.
    """
    db = next(get_db())  # Get a database session

    if request.content_type == 'application/json':
        data = request.get_json()  # Get the input JSON data
    else:
        data = {
            'is_tv_subscriber_pred': bool(request.form.get('is_tv_subscriber_pred')),
            'is_movie_package_subscriber_pred': bool(request.form.get('is_movie_package_subscriber_pred')),
            'subscription_age_pred': float(request.form.get('subscription_age_pred')),
            'bill_avg_pred': float(request.form.get('bill_avg_pred')),
            'reamining_contract_pred': float(request.form.get('reamining_contract_pred')),
            'service_failure_count_pred': int(request.form.get('service_failure_count_pred')),
            'download_avg_pred': float(request.form.get('download_avg_pred')),
            'upload_avg_pred': float(request.form.get('upload_avg_pred')),
            'download_over_limit_pred': int(request.form.get('download_over_limit_pred')),
            'model_choice': request.form.get('model_choice')
        }

    # Determine which model to use
    model_choice = data.get('model_choice', 'random_forest')
    if model_choice not in models:
        return jsonify({'error': 'Model not found'}), 404

    data_clear = data.copy()
    data_clear.pop('model_choice')

    # Preprocess input data
    processed_features = preprocess_input_data(data_clear)

    # Ensure processed_features is a numpy array with the right shape
    processed_features = np.array(processed_features).reshape(1, -1)

    # Create a DataFrame with the original column names
    processed_features_df = pd.DataFrame(processed_features,
                                         columns=['is_tv_subscriber', 'is_movie_package_subscriber',
                                                  'subscription_age', 'bill_avg', 'reamining_contract',
                                                  'service_failure_count', 'download_avg', 'upload_avg',
                                                  'download_over_limit'])

    prediction_prob = None
    model_used = model_choice

    # Make the prediction using the selected model
    try:
        if model_used == 'tensorflow_keras':
            prediction_prob = models[model_used].predict(processed_features_df)
            prediction_prob = prediction_prob.flatten()
            prediction_prob = prediction_prob[0]
        else:
            if hasattr(models[model_used], 'predict_proba'):
                prediction_prob = models[model_used].predict_proba(processed_features_df)[:, 1]
            else:
                prediction = models[model_used].predict(processed_features_df)
                prediction_prob = prediction.flatten()[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    response = {
        'model_used': model_used,
        'prediction_prob': round(float(prediction_prob), 2)
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

    if request.content_type == 'application/json':
        return jsonify(response), 200
    else:
        return render_template('result.html', prediction=response), 200


@route_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict the probability of churn for multiple customers using either the Random Forest or SVM model.

    The model to use is specified in the form input. The features for the predictions are extracted
    from the CSV file and used to make predictions. The results, along with the input data, are saved
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

    model_choice = request.form.get('model_choice', 'random_forest')
    if model_choice not in models:
        return jsonify({'error': 'Model not found'}), 404

    try:
        data = pd.read_csv(file)

        responses = []

        for index, row in data.iterrows():
            # Preprocess input data
            processed_features = preprocess_input_data(row)

            # Ensure processed_features is a numpy array with the right shape
            processed_features = np.array(processed_features).reshape(1, -1)

            # Create a DataFrame with the original column names
            processed_features_df = pd.DataFrame(processed_features,
                                                 columns=['is_tv_subscriber', 'is_movie_package_subscriber',
                                                          'subscription_age', 'bill_avg', 'reamining_contract',
                                                          'service_failure_count', 'download_avg', 'upload_avg',
                                                          'download_over_limit'])

            prediction_prob = None

            if model_choice == 'tensorflow_keras':
                prediction_prob = models[model_choice].predict(processed_features_df)
                prediction_prob = prediction_prob.flatten()
                prediction_prob = prediction_prob[0]
            else:
                if hasattr(models[model_choice], 'predict_proba'):
                    prediction_prob = models[model_choice].predict_proba(processed_features_df)[:, 1]
                else:
                    prediction = models[model_choice].predict(processed_features_df)
                    prediction_prob = prediction.flatten()[0]

            response = {
                'model_used': model_choice,
                'prediction_prob': round(float(prediction_prob), 2)
            }

            new_prediction_data = row.to_dict()
            new_prediction_data.update({
                'prediction_prob': prediction_prob,
                'model_used': model_choice
            })
            new_prediction = Prediction.from_dict(new_prediction_data)

            db.add(new_prediction)
            db.commit()
            db.refresh(new_prediction)

            responses.append(response)

        return jsonify(responses), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
