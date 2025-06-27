import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

db_file_path = os.getenv('database_file')

db_dir = os.path.dirname(db_file_path)
if not os.path.exists(db_dir):
    os.makedirs(db_dir)


conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,                   
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
""")

cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "test123"))
conn.commit()
conn.close()