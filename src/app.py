from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from src.config import config_by_name
from src.route import route_bp

db = SQLAlchemy()


def create_app(config_name: str) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    db.init_app(app)

    app.register_blueprint(route_bp)

    return app
