import sqlite3
import pandas as pd
from datetime import datetime
import tempfile
import os

class DatabaseManager:
    def __init__(self):
        self.db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = self.db_file.name
        self.init_database()
    
    def __del__(self):
        if hasattr(self, 'db_file'):
            self.db_file.close()
            os.unlink(self.db_path)
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY,
                    dataset_id INTEGER,
                    timestamp TIMESTAMP NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL,
                    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_measurements_dataset_timestamp 
                ON measurements(dataset_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_measurements_metric 
                ON measurements(metric_type)
            """)

    def store_dataset(self, filename: str, df: pd.DataFrame) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert datetime to ISO format string
            start_time = df['dt'].min().isoformat()
            end_time = df['dt'].max().isoformat()
            
            cursor.execute("""
                INSERT INTO datasets (filename, start_time, end_time)
                VALUES (?, ?, ?)
            """, (filename, start_time, end_time))
            
            dataset_id = cursor.lastrowid
            
            # Convert DataFrame to format suitable for measurements table
            data_to_insert = []
            for column in df.columns:
                if column != 'dt':
                    for _, row in df[['dt', column]].iterrows():
                        if pd.notna(row[column]):  # Skip NaN values
                            data_to_insert.append((
                                dataset_id,
                                row['dt'].isoformat(),  # Convert datetime to ISO format string
                                column,
                                float(row[column])
                            ))
            
            cursor.executemany("""
                INSERT INTO measurements (dataset_id, timestamp, metric_type, value)
                VALUES (?, ?, ?, ?)
            """, data_to_insert)
            
            return dataset_id
                
            cursor.executemany("""
                INSERT INTO measurements (dataset_id, timestamp, metric_type, value)
                VALUES (?, ?, ?, ?)
            """, data_to_insert)
            
            return dataset_id

    def get_available_metrics(self) -> list:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT metric_type FROM measurements")
            return [row[0] for row in cursor.fetchall()]

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
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=dataset_ids + metrics)
            # Käytetään ISO8601-formaattia aikaleiman parsimiseen
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
            pivoted = df.pivot_table(
                index=['filename', 'timestamp'],
                columns='metric_type',
                values='value'
            ).reset_index()
            return pivoted