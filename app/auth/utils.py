"""
hashing functions for authentication
"""

from werkzeug.security import check_password_hash, generate_password_hash

def hash_password(password):
    return generate_password_hash(password)

def check_password(hashed_password, password):
    return check_password_hash(hashed_password, password)
