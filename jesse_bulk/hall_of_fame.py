"""
DNA Hall of Fame - Persistent storage for best performing strategies

Stores and manages the best DNAs discovered across all jesse-bulk runs,
enabling long-term tracking, analysis, and cross-validation.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class HallOfFame:
    """Manages persistent storage of best performing DNAs"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Hall of Fame database
        
        Args:
            db_path: Path to SQLite database. Defaults to 'storage/hall_of_fame.db'
        """
        if db_path is None:
            db_path = Path("storage/hall_of_fame.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main DNA records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dna_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dna TEXT NOT NULL,
                dna_hash TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                discovery_date TIMESTAMP NOT NULL,
                discovery_source TEXT,
                selection_method TEXT,
                
                -- Key performance metrics
                sharpe_ratio REAL,
                net_profit_percentage REAL,
                win_rate REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                expectancy_percentage REAL,
                ratio_avg_win_loss REAL,
                calmar_ratio REAL,
                sortino_ratio REAL,
                omega_ratio REAL,
                
                -- Test conditions
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                backtest_days INTEGER,
                
                -- Additional metrics (JSON)
                full_metrics TEXT,
                hyperparameters TEXT,
                
                -- Meta information
                notes TEXT,
                tags TEXT,
                
                UNIQUE(dna_hash, strategy_name, symbol, timeframe)
            )
        """)
        
        # Performance history table for tracking over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dna_record_id INTEGER NOT NULL,
                test_date TIMESTAMP NOT NULL,
                test_type TEXT,  -- 'refinement', 'hall_of_fame_test', 'manual'
                
                -- Test parameters
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                
                -- Results
                sharpe_ratio REAL,
                net_profit_percentage REAL,
                win_rate REAL,
                max_drawdown REAL,
                finishing_balance REAL,
                
                -- Full results (JSON)
                full_results TEXT,
                
                FOREIGN KEY (dna_record_id) REFERENCES dna_records(id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dna_hash ON dna_records(dna_hash)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy ON dna_records(strategy_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_performance ON dna_records(
                sharpe_ratio DESC, net_profit_percentage DESC
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_dna(self, dna: str, metrics: Dict, strategy_name: str, 
                test_conditions: Dict, source: str = "refine-best", 
                selection_method: str = None, notes: str = None) -> int:
        """
        Add a new DNA to the hall of fame
        
        Args:
            dna: Base64 encoded DNA string
            metrics: Performance metrics dictionary
            strategy_name: Name of the strategy
            test_conditions: Dict with symbol, timeframe, dates, etc.
            source: Where this DNA was discovered
            selection_method: How it was selected (e.g., 'conservative', 'aggressive')
            notes: Optional notes about this DNA
            
        Returns:
            ID of the inserted record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generate hash for deduplication
        dna_hash = hashlib.sha256(dna.encode()).hexdigest()[:16]
        
        # Extract hyperparameters from DNA
        try:
            import base64
            hp_dict = json.loads(base64.b64decode(dna))
        except:
            hp_dict = {}
        
        # Calculate backtest days
        if 'start_date' in test_conditions and 'end_date' in test_conditions:
            try:
                # Handle various date formats
                start_str = test_conditions['start_date']
                end_str = test_conditions['end_date']
                
                # If the date string is just YYYY-MM-DD, pandas handles it fine
                # If it has time components or other formats, try to parse
                start = pd.to_datetime(start_str, errors='coerce')
                end = pd.to_datetime(end_str, errors='coerce')
                
                if pd.isna(start) or pd.isna(end):
                    # Try extracting just the date part if it's in the format YYYY-MM-DD
                    import re
                    date_pattern = r'(\d{4}-\d{2}-\d{2})'
                    start_match = re.search(date_pattern, start_str)
                    end_match = re.search(date_pattern, end_str)
                    
                    if start_match and end_match:
                        start = pd.to_datetime(start_match.group(1))
                        end = pd.to_datetime(end_match.group(1))
                    else:
                        backtest_days = None
                else:
                    backtest_days = (end - start).days
            except:
                backtest_days = None
        else:
            backtest_days = None
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO dna_records (
                    dna, dna_hash, strategy_name, discovery_date, discovery_source,
                    selection_method, sharpe_ratio, net_profit_percentage, win_rate,
                    max_drawdown, total_trades, expectancy_percentage, ratio_avg_win_loss,
                    calmar_ratio, sortino_ratio, omega_ratio, symbol, timeframe,
                    start_date, end_date, backtest_days, full_metrics, hyperparameters,
                    notes, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dna, dna_hash, strategy_name, datetime.now(), source,
                selection_method,
                metrics.get('sharpe_ratio'),
                metrics.get('net_profit_percentage'),
                metrics.get('win_rate'),
                metrics.get('max_drawdown'),
                metrics.get('total'),
                metrics.get('expectancy_percentage'),
                metrics.get('ratio_avg_win_loss'),
                metrics.get('calmar_ratio'),
                metrics.get('sortino_ratio'),
                metrics.get('omega_ratio'),
                test_conditions.get('symbol'),
                test_conditions.get('timeframe'),
                test_conditions.get('start_date'),
                test_conditions.get('end_date'),
                backtest_days,
                json.dumps(metrics),
                json.dumps(hp_dict),
                notes,
                None  # tags for future use
            ))
            
            record_id = cursor.lastrowid
            
            # Also add to performance history
            cursor.execute("""
                INSERT INTO performance_history (
                    dna_record_id, test_date, test_type, symbol, timeframe,
                    start_date, end_date, sharpe_ratio, net_profit_percentage,
                    win_rate, max_drawdown, finishing_balance, full_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id, datetime.now(), source,
                test_conditions.get('symbol'),
                test_conditions.get('timeframe'),
                test_conditions.get('start_date'),
                test_conditions.get('end_date'),
                metrics.get('sharpe_ratio'),
                metrics.get('net_profit_percentage'),
                metrics.get('win_rate'),
                metrics.get('max_drawdown'),
                metrics.get('finishing_balance'),
                json.dumps(metrics)
            ))
            
            conn.commit()
            return record_id
            
        except sqlite3.IntegrityError:
            # DNA already exists, update if better performance
            existing = cursor.execute("""
                SELECT id, sharpe_ratio FROM dna_records 
                WHERE dna_hash = ? AND strategy_name = ? 
                AND symbol = ? AND timeframe = ?
            """, (dna_hash, strategy_name, test_conditions.get('symbol'), 
                  test_conditions.get('timeframe'))).fetchone()
            
            if existing and metrics.get('sharpe_ratio', 0) > (existing[1] or 0):
                # Update with better performance
                cursor.execute("""
                    UPDATE dna_records SET
                        sharpe_ratio = ?, net_profit_percentage = ?, win_rate = ?,
                        max_drawdown = ?, full_metrics = ?, discovery_date = ?
                    WHERE id = ?
                """, (
                    metrics.get('sharpe_ratio'),
                    metrics.get('net_profit_percentage'),
                    metrics.get('win_rate'),
                    metrics.get('max_drawdown'),
                    json.dumps(metrics),
                    datetime.now(),
                    existing[0]
                ))
                conn.commit()
                return existing[0]
                
        finally:
            conn.close()
    
    def get_best_dnas(self, strategy_name: Optional[str] = None,
                      metric: str = 'sharpe_ratio', 
                      limit: int = 10,
                      min_trades: Optional[int] = None,
                      symbol: Optional[str] = None,
                      timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieve best performing DNAs
        
        Args:
            strategy_name: Filter by strategy name
            metric: Metric to sort by
            limit: Number of results to return
            min_trades: Minimum number of trades filter
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            
        Returns:
            DataFrame with best DNAs
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT * FROM dna_records 
            WHERE 1=1
        """
        params = []
        
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        
        if min_trades:
            query += " AND total_trades >= ?"
            params.append(min_trades)
            
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
            
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        
        query += f" ORDER BY {metric} DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Parse JSON fields
        if not df.empty:
            df['hyperparameters'] = df['hyperparameters'].apply(lambda x: json.loads(x) if x else {})
            df['full_metrics'] = df['full_metrics'].apply(lambda x: json.loads(x) if x else {})
        
        return df
    
    def add_performance_test(self, dna_hash: str, test_results: Dict,
                           test_conditions: Dict, test_type: str = "hall_of_fame_test"):
        """Add a new performance test result for an existing DNA"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find the DNA record
        record = cursor.execute(
            "SELECT id FROM dna_records WHERE dna_hash = ?", (dna_hash,)
        ).fetchone()
        
        if not record:
            conn.close()
            raise ValueError(f"DNA with hash {dna_hash} not found")
        
        cursor.execute("""
            INSERT INTO performance_history (
                dna_record_id, test_date, test_type, symbol, timeframe,
                start_date, end_date, sharpe_ratio, net_profit_percentage,
                win_rate, max_drawdown, finishing_balance, full_results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record[0], datetime.now(), test_type,
            test_conditions.get('symbol'),
            test_conditions.get('timeframe'),
            test_conditions.get('start_date'),
            test_conditions.get('end_date'),
            test_results.get('sharpe_ratio'),
            test_results.get('net_profit_percentage'),
            test_results.get('win_rate'),
            test_results.get('max_drawdown'),
            test_results.get('finishing_balance'),
            json.dumps(test_results)
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_history(self, dna_hash: str) -> pd.DataFrame:
        """Get all performance history for a specific DNA"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT ph.*, dr.strategy_name, dr.dna
            FROM performance_history ph
            JOIN dna_records dr ON ph.dna_record_id = dr.id
            WHERE dr.dna_hash = ?
            ORDER BY ph.test_date DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[dna_hash])
        conn.close()
        
        return df
    
    def get_statistics(self) -> Dict:
        """Get overall statistics about the hall of fame"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total DNAs
        stats['total_dnas'] = cursor.execute(
            "SELECT COUNT(*) FROM dna_records"
        ).fetchone()[0]
        
        # DNAs by strategy
        stats['by_strategy'] = dict(cursor.execute(
            "SELECT strategy_name, COUNT(*) FROM dna_records GROUP BY strategy_name"
        ).fetchall())
        
        # Average metrics
        avg_metrics = cursor.execute("""
            SELECT 
                AVG(sharpe_ratio) as avg_sharpe,
                AVG(net_profit_percentage) as avg_profit,
                AVG(win_rate) as avg_win_rate,
                AVG(max_drawdown) as avg_drawdown
            FROM dna_records
        """).fetchone()
        
        stats['average_metrics'] = {
            'sharpe_ratio': avg_metrics[0],
            'net_profit_percentage': avg_metrics[1],
            'win_rate': avg_metrics[2],
            'max_drawdown': avg_metrics[3]
        }
        
        # Top performers
        stats['top_sharpe'] = cursor.execute(
            "SELECT MAX(sharpe_ratio) FROM dna_records"
        ).fetchone()[0]
        
        stats['top_profit'] = cursor.execute(
            "SELECT MAX(net_profit_percentage) FROM dna_records"
        ).fetchone()[0]
        
        # Total performance tests
        stats['total_tests'] = cursor.execute(
            "SELECT COUNT(*) FROM performance_history"
        ).fetchone()[0]
        
        conn.close()
        return stats
    
    def export_to_csv(self, output_path: str, strategy_name: Optional[str] = None):
        """Export hall of fame to CSV file"""
        df = self.get_best_dnas(strategy_name=strategy_name, limit=10000)
        
        # Flatten hyperparameters for CSV
        if not df.empty:
            hp_df = pd.json_normalize(df['hyperparameters'])
            hp_df.columns = [f'hp_{col}' for col in hp_df.columns]
            df = pd.concat([df.drop('hyperparameters', axis=1), hp_df], axis=1)
            df = df.drop('full_metrics', axis=1)  # Too large for CSV
        
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} DNAs to {output_path}")


def get_hall_of_fame(db_path: Optional[str] = None) -> HallOfFame:
    """Get or create a Hall of Fame instance"""
    return HallOfFame(db_path)