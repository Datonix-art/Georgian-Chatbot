from flask import Flask
from flask_login import LoginManager
from flask_cors import CORS
from app.models import User

login = LoginManager()

def create_app():
    # flask interface
    app = Flask(__name__)

    # flask configs
    app.config.from_object('app.config.Config')

    # flask login settings
    login.init_app(app)
    login.login_view = 'login'
     
    @login.user_loader
    def load_user(user_id):
        return User.get(user_id)
     
    # flask cors
    CORS(app)

    return app