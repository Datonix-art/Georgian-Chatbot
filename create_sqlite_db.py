import os
import sqlite3

database_dir = os.path.join(os.getcwd(), 'database')

if not os.path.exists(database_dir):
    os.makedirs(database_dir)

database_file_path = os.path.join(database_dir, 'database.db')

conn = sqlite3.connect(database_file_path)
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