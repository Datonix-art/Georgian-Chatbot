from flask import Flask, render_template
from flask_login import LoginManager
from flask_cors import CORS
from app.auth.models import User
import os

login = LoginManager()

"""App creation interface"""
def create_app():
    # flask interface
    app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'app', 'templates'))
    
    # flask blueprints
    from app.auth.routes import auth_bp
    from app.bot.routes import bot_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(bot_bp)

    # flask configs
    app.config.from_object('app.config.Config')

    # flask login settings
    login.init_app(app)
    login.login_view = 'login'
    
    # error handler routes
    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def error_500(error):
        return render_template('500.html'), 500
    
    # user loader
    @login.user_loader
    def load_user(user_id):
        return User.get(user_id)
     
    # flask cors lib
    CORS(app)

    return app