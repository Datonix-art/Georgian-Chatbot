from flask_login import login_required
from flask import render_template

from app import create_app

app = create_app()

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/create-account')
def create_account():
    return render_template('signup.html')

@app.route('/log-in')
def login():
    return render_template('login.html')

