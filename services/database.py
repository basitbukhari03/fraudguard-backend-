"""
Database Service
-----------------
MongoDB Atlas connection helper using pymongo.
"""

import os
from pymongo import MongoClient

# Connection string from environment variable
MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://basitbukhari03:basit0112233@cluster0.nj9xhhy.mongodb.net/?appName=Cluster0"
)

# Create client and select database
client = MongoClient(MONGO_URI)
db = client["fraudguard"]

# Collections
users_collection = db["users"]

# Create unique index on email to prevent duplicate signups
users_collection.create_index("email", unique=True)
