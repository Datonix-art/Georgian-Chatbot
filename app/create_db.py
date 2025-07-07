import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

db_file_path = os.getenv('database_file')

os.makedirs(os.path.dirname(db_file_path), exist_ok=True)

conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Users model to store authenticated users data
cursor.execute("""
    CREATE TABLE IF NOT EXISTS User (
        id INTEGER PRIMARY KEY AUTOINCREMENT,                   
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )
""")

# insert admin user into Users table
cursor.execute("INSERT INTO User (username, password) VALUES (?, ?)", ("admin", "test123"))

# Conversation model to store Q&A  history
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Conversation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        context_used TEXT NOT NULL,
        success BOOLEAN NOT NULL DEFAULT 0,
        timestamp INTEGER NOT NULL,
        response_time REAL NOT NULL,
        FOREIGN KEY (user_id) REFERENCES User(id) ON DELETE CASCADE
    )    
""")

# Feedback model for user ratings
cursor.execute("""
    CREATE TABLE IF NOT EXISTS Feedback (
        id PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        conversation_id INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        comment TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        FOREIGN KEY (user_id) REFERENCES User(id) ON DELETE CASCADE,
        FOREIGN KEY (conversation_id) REFERENCES Conversation(id) ON DELETE CASCADE          
    )
""")


conn.commit()
conn.close()
