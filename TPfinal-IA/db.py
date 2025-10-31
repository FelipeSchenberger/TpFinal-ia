import sqlite3
import numpy as np

DB_NAME = "faces.db"

def create_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces (
                    name TEXT,
                    vector BLOB
                )''')
    conn.commit()
    conn.close()

def insert_face(name, vector):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO faces (name, vector) VALUES (?, ?)", 
              (name, vector.astype(np.uint8).tobytes()))
    conn.commit()
    conn.close()

def load_faces():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT name, vector FROM faces")
    data = c.fetchall()
    conn.close()

    names = []
    vectors = []
    for row in data:
        names.append(row[0])
        vectors.append(np.frombuffer(row[1], dtype=np.uint8))
    return names, vectors
