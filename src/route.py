from flask import Blueprint, jsonify, request
from src.database.models import Prediction
from src.database.db import get_db

route_bp = Blueprint('route', __name__)


@route_bp.route('/test', methods=['GET'])
def test():
    return jsonify(
        {'message': 'Hello World!!!'}
    ), 200


@route_bp.route('/predictions', methods=['POST', 'GET'])
def predictions():
    db = next(get_db())
    if request.method == 'GET':
        predictions = db.query(Prediction).all()
        return jsonify([prediction.to_dict() for prediction in predictions]), 200

    if request.method == 'POST':
        data = request.get_json()

        predictions = Prediction(**data)

        db.add(predictions)
        db.commit()
        db.refresh(predictions)

        return jsonify(predictions.to_dict()), 201
