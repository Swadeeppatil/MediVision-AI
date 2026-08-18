import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='fracture_scans.db'):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_database(self):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS scans
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     patient_name TEXT,
                     patient_age INTEGER,
                     patient_gender TEXT,
                     fracture_type TEXT,
                     confidence REAL,
                     severity TEXT,
                     timestamp DATETIME,
                     image_path TEXT,
                     highlighted_path TEXT,
                     thermal_path TEXT)''')
        conn.commit()
        conn.close()

    def save_scan_result(self, patient_name, patient_age, patient_gender, fracture_type, confidence, severity, image_path, highlighted_path=None, thermal_path=None):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""INSERT INTO scans 
                    (patient_name, patient_age, patient_gender, fracture_type, confidence, severity, timestamp, image_path, highlighted_path, thermal_path) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 (patient_name, patient_age, patient_gender, fracture_type, confidence, severity, datetime.now(), image_path, highlighted_path, thermal_path))
        conn.commit()
        conn.close()

    def fetch_recent_scans(self, limit=10):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""SELECT timestamp, patient_name, patient_age, patient_gender, fracture_type, severity 
                    FROM scans 
                    ORDER BY timestamp DESC 
                    LIMIT ?""", (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    
    def get_scan_by_id(self, scan_id):
        conn = self.get_connection()
        c = conn.cursor()
        c.execute("""SELECT * FROM scans WHERE id = ?""", (scan_id,))
        row = c.fetchone()
        conn.close()
        return row
