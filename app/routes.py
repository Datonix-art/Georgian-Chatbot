from flask import render_template, request, flash, redirect, url_for
from flask_login import login_user, logout_user
from dotenv import load_dotenv
import sqlite3
import os
from app import create_app
from app.models import User
from app.utils import hash_password, check_password

load_dotenv()

database_file = os.getenv("database_file")

app = create_app()

"""Main page route"""
@app.route('/')
def index():
    """main chat interface"""
    return render_template('main.html')

""" Authentication routes """

@app.route('/create-account')
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=["POST"])
def create_account():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username, ))
    account_exists = cursor.fetchone()

    if account_exists:
        conn.close()
        flash('Account already exists with this username')
        return redirect(url_for('signup'))
    
    hashed_password = hash_password(password)

    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()
    
    return redirect(url_for('login'))

@app.route('/log-in')
def login():
    return render_template('login.html')

@app.route('/log-in-account', methods=["POST"])
def login_account():
    data = request.form
    username = data.get('username')
    password = data.get('password')

    user = User.find_by_username(username)
    
    if not user:
        flash("Account doesn't exist with such Username")
        return redirect(url_for('signup'))
    
    if not check_password(user.hashed_password, password):
        flash("Incorrect password.")
        return redirect(url_for('login'))
    
    login_user(user)
    flash("Successfully logged in!")
    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))


"""Chatbot routes"""

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    pass