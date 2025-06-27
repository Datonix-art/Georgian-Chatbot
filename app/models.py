import sqlite3
from flask_login import UserMixin
import os

database_file =  os.path.join(os.getcwd(), 'database', 'database.db')

class User(UserMixin):
    def __init__(self, id_, username):
        self.id = id_
        self.username = username

    @staticmethod
    def get(user_id):
        conn = sqlite3.connect(database_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id, ))
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(*row)
        return None
    
