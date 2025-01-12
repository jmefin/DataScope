import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import threading

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance.conn = sqlite3.connect(
                        ':memory:',
                        check_same_thread=False
                    )
                    cls._instance.init_database()
        return cls._instance
    
    def __del__(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
    
    def init_database(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                start_time TIMESTAMP,
                end_time TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY,
                dataset_id INTEGER,
                timestamp TIMESTAMP NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_dataset_timestamp 
            ON measurements(dataset_id, timestamp)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_measurements_metric 
            ON measurements(metric_type)
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metric_thresholds (
                metric_type TEXT PRIMARY KEY,
                optimal_min REAL,
                optimal_max REAL,
                alarm_min REAL,
                alarm_max REAL
            )
        """)

    def store_dataset(self, filename: str, df: pd.DataFrame) -> int:
        cursor = self.conn.cursor()
        
        # Convert datetime to ISO format string
        start_time = df['timestamp'].min().isoformat()
        end_time = df['timestamp'].max().isoformat()
        
        cursor.execute("""
            INSERT INTO datasets (filename, start_time, end_time)
            VALUES (?, ?, ?)
        """, (filename, start_time, end_time))
        
        dataset_id = cursor.lastrowid
        
        # Convert DataFrame to format suitable for measurements table
        data_to_insert = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            for _, row in df[['timestamp', column]].iterrows():
                if pd.notna(row[column]):  # Skip NaN values
                    data_to_insert.append((
                        dataset_id,
                        row['timestamp'].isoformat(),  # Convert datetime to ISO format string
                        column,
                        float(row[column])
                    ))
        
        cursor.executemany("""
            INSERT INTO measurements (dataset_id, timestamp, metric_type, value)
            VALUES (?, ?, ?, ?)
        """, data_to_insert)
        
        return dataset_id

    def get_available_metrics(self) -> list:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT metric_type FROM measurements")
        return [row[0] for row in cursor.fetchall()]

    def set_metric_thresholds(self, metric_type: str, 
                            optimal_min: float, optimal_max: float,
                            alarm_min: float, alarm_max: float):
        """Set thresholds for a metric type"""
        self.conn.execute("""
            INSERT OR REPLACE INTO metric_thresholds 
            (metric_type, optimal_min, optimal_max, alarm_min, alarm_max)
            VALUES (?, ?, ?, ?, ?)
        """, (metric_type, optimal_min, optimal_max, alarm_min, alarm_max))
        self.conn.commit()

    def get_metric_thresholds(self, metric_type: str) -> dict:
        """Get thresholds for a metric type"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT optimal_min, optimal_max, alarm_min, alarm_max
            FROM metric_thresholds
            WHERE metric_type = ?
        """, (metric_type,))
        result = cursor.fetchone()
        if result:
            return {
                'optimal_min': result[0],
                'optimal_max': result[1],
                'alarm_min': result[2],
                'alarm_max': result[3]
            }
        return None

    def get_dataset_data(self, dataset_ids: list, metrics: list) -> pd.DataFrame:
        placeholders = ','.join('?' * len(dataset_ids))
        metric_placeholders = ','.join('?' * len(metrics))
        
        query = f"""
            SELECT d.filename, m.timestamp, m.metric_type, m.value
            FROM measurements m
            JOIN datasets d ON m.dataset_id = d.id
            WHERE d.id IN ({placeholders}) 
            AND m.metric_type IN ({metric_placeholders})
            ORDER BY m.timestamp
        """
        
        df = pd.read_sql_query(query, self.conn, params=dataset_ids + metrics)
        # Convert timestamp to datetime using ISO8601 format
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
        pivoted = df.pivot_table(
            index=['filename', 'timestamp'],
            columns='metric_type',
            values='value'
        ).reset_index()
        return pivoted
