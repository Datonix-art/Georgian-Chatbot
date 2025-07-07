import os
import sqlite3
from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_user, logout_user
from dotenv import load_dotenv
from app.auth.models import User
from app.auth.utils import hash_password, check_password

load_dotenv()

database_file = os.getenv("database_file")

auth_bp = Blueprint('auth_bp', __name__)

""" create user"""
@auth_bp.route('/create-account', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        data = request.form
        username = data.get('username').strip()
        password = data.get('password').strip()
        try: 
            with sqlite3.connect(database_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM User WHERE username = ?", (username, ))
                account_exists = cursor.fetchone()
                
                if account_exists:
                    conn.close()
                    flash('აქაუნთი ამ სახელით უკვე არსებობს. ცადეთ სხვა სახელის გამოყენება')
                    return redirect(request.url)
            
                hashed_password = hash_password(password)

                cursor.execute("INSERT INTO User (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
                flash('წარმატებით შექმენით აქაუნთი. გთხოვთ დარეგისტრირდით')
                return redirect(url_for('auth_bp.login'))
        except sqlite3.DatabaseError as e:
            print(f'Error connecting to database: {e}')
            return redirect(url_for('auth_bp.signup'))
    return render_template('signup.html')

""" Login user"""
@auth_bp.route('/log-in', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        username = data.get('username').strip()
        password = data.get('password').strip()
        if not username or not password:
            flash("მომხმარებლის სახელი და პაროლი სავალდებულოა")
            return redirect(request.url)

        user = User.find_by_username(username)
        
        if not user:
            flash("აქაუნთი ამ მომხარებლის სახელით არ არსებობს")
            return redirect(request.url)
        
        if not check_password(user.hashed_password, password):
            flash("პაროლი არასწორია")
            return redirect(request.url)
        
        login_user(user)
        flash("წარმატებით დარეგისტრირდით")
        return redirect(url_for('bot_bp.index'))
    return render_template('login.html')

""" Logout user """
@auth_bp.route('/logout')
def logout():
    logout_user()
    flash('წარმატებით გამოხვედით აქაუნთიდან')
    return redirect(url_for('bot_bp.index'))
