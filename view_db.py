import os, sqlite3
from prettytable import PrettyTable

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')

print("Checking database at:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
if not tables:
    print("❌ No tables found.")
else:
    print("Tables:", [t[0] for t in tables])

cursor.execute("SELECT * FROM students;")
rows = cursor.fetchall()

if not rows:
    print("No records found in students table.")
else:
    table = PrettyTable(["ID", "Name", "Email", "Department", "Image Path", "Emotion", "Timestamp"])
    for row in rows:
        table.add_row(row)
    print(table)

conn.close()
