from flask import Flask
from flask_login import LoginManager
from flask_cors import CORS

login = LoginManager()

def create_app():
    app = Flask(__name__)

    app.config.from_object('app.config.Config')

    login.init_app(app)
    CORS(app)
    return app