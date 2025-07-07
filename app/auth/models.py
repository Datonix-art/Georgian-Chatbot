from flask_login import UserMixin
import sqlite3
import os

database_file = os.getenv('database_file')

class User(UserMixin):
    def __init__(self, id_, username, hashed_password=None):
        self.id = id_
        self.username = username
        self.hashed_password = hashed_password

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM User WHERE id = ?", (user_id, ))
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(*row)  # id, username, hashed_password
        return None

    @staticmethod
    def find_by_username(username):
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password FROM User WHERE username = ?", (username, ))
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(*row)
        return None

    
    
    
    