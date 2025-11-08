import os
import sqlite3
from prettytable import PrettyTable

# Always connect to the correct database file in your project folder
DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')

# Debug info to confirm path being read
print(f"📍 Checking database at: {DB_PATH}")

# Connect to database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Check if table exists
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
if not tables:
    print("❌ No tables found.")
else:
    print("📦 Tables:", [t[0] for t in tables])

# Show data if table exists
cursor.execute("SELECT * FROM students;")
rows = cursor.fetchall()

if not rows:
    print("📭 No records found in students table.")
else:
    table = PrettyTable(["ID", "Name", "Email", "Department", "Image Path", "Emotion", "Timestamp"])
    for row in rows:
        table.add_row(row)
    print(table)

conn.close()
