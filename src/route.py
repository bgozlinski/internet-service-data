from flask import Blueprint, jsonify

route_bp = Blueprint('route', __name__)


@route_bp.route('/test', methods=['GET'])
def test():
    return jsonify(
        {'message': 'Hello World!!!'}
    ), 200

