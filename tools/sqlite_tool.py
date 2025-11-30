"""
Azmy Retail Analytics Copilot
SQLite tool for database access and schema introspection
"""
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging


class SQLiteTool:
    """Tool for executing SQL queries and introspecting database schema"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def get_schema(self) -> Dict[str, List[str]]:
        """Get database schema information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                schema = {}
                for table in tables:
                    # Handle table names with spaces by quoting them
                    table_query = f'PRAGMA table_info("{table}")'
                    cursor.execute(table_query)
                    columns = [row[1] for row in cursor.fetchall()]
                    schema[table] = columns
                
                return schema
        except Exception as e:
            self.logger.error(f"Error getting schema: {e}")
            return {}
    
    def execute_query(self, query: str) -> Tuple[bool, List[Dict[str, Any]], str]:
        """
        Execute SQL query and return results
        Returns: (success, results, error_message)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = [dict(row) for row in rows]
                
                return True, results, ""
                
        except Exception as e:
            error_msg = f"SQL Error: {str(e)}"
            self.logger.error(error_msg)
            return False, [], error_msg
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample rows from a table"""
        # Quote table name to handle spaces
        quoted_table = f'"{table_name}"'
        query = f"SELECT * FROM {quoted_table} LIMIT {limit}"
        success, results, _ = self.execute_query(query)
        return results if success else []
    
    def get_table_count(self, table_name: str) -> int:
        """Get row count for a table"""
        # Quote table name to handle spaces
        quoted_table = f'"{table_name}"'
        query = f"SELECT COUNT(*) as count FROM {quoted_table}"
        success, results, _ = self.execute_query(query)
        return results[0]['count'] if success and results else 0
    
    def validate_query_syntax(self, query: str) -> Tuple[bool, str]:
        """Validate SQL query syntax without executing"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                return True, ""
        except Exception as e:
            return False, str(e)
    
    def get_schema_description(self) -> str:
        """Get a human-readable description of the database schema"""
        schema = self.get_schema()
        description = "Database Schema:\n"
        
        for table, columns in schema.items():
            description += f"\n{table}:\n"
            for col in columns:
                description += f"  - {col}\n"
            
            # Add sample data
            sample = self.get_table_sample(table, 2)
            if sample:
                description += f"  Sample: {sample[0]}\n"
        
        return description
