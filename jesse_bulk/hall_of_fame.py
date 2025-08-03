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
from typing import Dict, List, Optional, Tuple, Any
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
        
        # Add new columns if they don't exist (for backward compatibility)
        try:
            cursor.execute("ALTER TABLE dna_records ADD COLUMN wins INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE dna_records ADD COLUMN win_score REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE dna_records ADD COLUMN last_win_date TIMESTAMP")
            conn.commit()
        except sqlite3.OperationalError:
            # Columns already exist
            pass
        
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
    
    def prune_by_performance(self, min_success_rate: float = 90.0,
                           min_avg_balance: float = 100000.0,
                           min_lowest_balance: float = 5000.0,
                           min_avg_profit: float = 200.0,
                           require_wins: bool = False,
                           dry_run: bool = True) -> Dict[str, Any]:
        """
        Remove DNAs that don't meet performance criteria based on their test history
        
        Args:
            min_success_rate: Minimum success rate percentage (default: 90%)
            min_avg_balance: Minimum average finishing balance (default: $100k)
            min_lowest_balance: Minimum lowest finishing balance (default: $5k)
            min_avg_profit: Minimum average profit percentage (default: 200%)
            require_wins: If True, remove DNAs with zero wins (default: False)
            dry_run: If True, only show what would be removed without deleting
            
        Returns:
            Dictionary with pruning statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all DNAs with their performance statistics
        query = """
            SELECT 
                dr.id,
                dr.dna,
                dr.dna_hash,
                dr.strategy_name,
                dr.sharpe_ratio as original_sharpe,
                dr.net_profit_percentage as original_profit,
                dr.wins,
                COUNT(ph.id) as total_tests,
                SUM(CASE WHEN ph.sharpe_ratio IS NOT NULL THEN 1 ELSE 0 END) as successful_tests,
                AVG(CASE WHEN ph.finishing_balance IS NOT NULL THEN ph.finishing_balance ELSE NULL END) as avg_balance,
                MIN(CASE WHEN ph.finishing_balance IS NOT NULL THEN ph.finishing_balance ELSE NULL END) as min_balance,
                MAX(CASE WHEN ph.finishing_balance IS NOT NULL THEN ph.finishing_balance ELSE NULL END) as max_balance,
                AVG(CASE WHEN ph.net_profit_percentage IS NOT NULL THEN ph.net_profit_percentage ELSE NULL END) as avg_profit
            FROM dna_records dr
            LEFT JOIN performance_history ph ON dr.id = ph.dna_record_id
                AND ph.test_type IN ('hall_of_fame_test', 'refine-best')
            GROUP BY dr.id, dr.dna, dr.dna_hash, dr.strategy_name, dr.wins
        """
        
        df = pd.read_sql_query(query, conn)
        
        # Calculate success rate
        df['success_rate'] = (df['successful_tests'] / df['total_tests'] * 100).fillna(0)
        
        # DNAs with no test history
        no_tests = df[df['total_tests'] == 0]
        
        # Apply filters
        conditions = (df['total_tests'] > 0) & (  # Has been tested
            (df['success_rate'] < min_success_rate) |
            (df['avg_balance'] < min_avg_balance) |
            (df['min_balance'] < min_lowest_balance) |
            (df['avg_profit'] < min_avg_profit)
        )
        
        # Add wins requirement if specified
        if require_wins:
            # Include DNAs with zero wins in removal list
            conditions = conditions | ((df['wins'].isna()) | (df['wins'] == 0))
        
        to_remove = df[conditions]
        
        # DNAs that pass all criteria
        keep_conditions = (
            (df['total_tests'] > 0) &
            (df['success_rate'] >= min_success_rate) &
            (df['avg_balance'] >= min_avg_balance) &
            (df['min_balance'] >= min_lowest_balance) &
            (df['avg_profit'] >= min_avg_profit)
        )
        
        if require_wins:
            # Must also have at least one win
            keep_conditions = keep_conditions & (df['wins'] > 0)
        
        to_keep = df[keep_conditions]
        
        stats = {
            'total_dnas': len(df),
            'no_test_history': len(no_tests),
            'to_remove': len(to_remove),
            'to_keep': len(to_keep),
            'criteria': {
                'min_success_rate': min_success_rate,
                'min_avg_balance': min_avg_balance,
                'min_lowest_balance': min_lowest_balance,
                'min_avg_profit': min_avg_profit
            }
        }
        
        print("\n🧹 HALL OF FAME PRUNING ANALYSIS")
        print("="*60)
        print(f"Total DNAs: {stats['total_dnas']}")
        print(f"DNAs with no test history: {stats['no_test_history']}")
        print(f"DNAs to remove: {stats['to_remove']}")
        print(f"DNAs to keep: {stats['to_keep']}")
        
        print(f"\nRemoval Criteria:")
        print(f"  - Success Rate < {min_success_rate}%")
        print(f"  - Avg Balance < ${min_avg_balance:,.2f}")
        print(f"  - Lowest Balance < ${min_lowest_balance:,.2f}")
        print(f"  - Avg Profit < {min_avg_profit}%")
        if require_wins:
            print(f"  - Zero wins (never won in any test)")
        
        if len(to_remove) > 0:
            print("\nDNAs to be removed:")
            for _, row in to_remove.iterrows():
                print(f"\n  ...{row['dna_hash'][-8:]} ({row['strategy_name']})")
                print(f"    - Wins: {int(row['wins']) if pd.notna(row['wins']) else 0}")
                print(f"    - Success Rate: {row['success_rate']:.0f}%")
                print(f"    - Avg Balance: ${row['avg_balance']:,.2f}")
                print(f"    - Min Balance: ${row['min_balance']:,.2f}")
                print(f"    - Avg Profit: {row['avg_profit']:.1f}%")
        
        if len(to_keep) > 0:
            print("\nTop performers to keep:")
            for _, row in to_keep.head(5).iterrows():
                print(f"\n  ...{row['dna_hash'][-8:]} ({row['strategy_name']})")
                print(f"    - Wins: {int(row['wins']) if pd.notna(row['wins']) else 0}")
                print(f"    - Success Rate: {row['success_rate']:.0f}%")
                print(f"    - Avg Balance: ${row['avg_balance']:,.2f}")
                print(f"    - Min Balance: ${row['min_balance']:,.2f}")
                print(f"    - Avg Profit: {row['avg_profit']:.1f}%")
        
        if not dry_run and len(to_remove) > 0:
            # Delete the DNAs and their performance history
            dna_ids = to_remove['id'].tolist()
            placeholders = ','.join('?' * len(dna_ids))
            
            cursor.execute(f"DELETE FROM performance_history WHERE dna_record_id IN ({placeholders})", dna_ids)
            cursor.execute(f"DELETE FROM dna_records WHERE id IN ({placeholders})", dna_ids)
            
            conn.commit()
            print(f"\n✅ Removed {len(to_remove)} DNAs from Hall of Fame")
        elif dry_run:
            print("\n⚠️  DRY RUN - No changes made. Run with dry_run=False to actually remove DNAs")
        
        conn.close()
        return stats


    def award_wins(self, test_results_df: pd.DataFrame, test_type: str = "hall_of_fame_test", 
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Award wins to DNAs based on their performance in a test batch
        
        Uses a weighted composite score:
        - 30% Success Rate (percentage of runs without errors)
        - 25% Average Profit 
        - 20% Average Sharpe Ratio
        - 15% Average Finishing Balance
        - 10% Minimum Balance (risk control)
        
        You can customize weights by passing a weights dict:
        weights = {
            'success_rate': 0.40,  # Prioritize reliability
            'avg_profit': 0.20,
            'avg_sharpe': 0.20,
            'avg_balance': 0.10,
            'min_balance': 0.10
        }
        
        Args:
            test_results_df: DataFrame with test results including 'dna' column
            test_type: Type of test for tracking
            
        Returns:
            Dictionary with win statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Group by DNA and calculate statistics
        dna_stats = []
        for dna_str in test_results_df['dna'].unique():
            dna_results = test_results_df[test_results_df['dna'] == dna_str]
            
            # Calculate success rate
            if 'sharpe_ratio' in dna_results.columns:
                successful_runs = dna_results['sharpe_ratio'].notna().sum()
                total_runs = len(dna_results)
                success_rate = (successful_runs / total_runs) if total_runs > 0 else 0
            else:
                success_rate = 0
            
            # Calculate averages (only for successful runs)
            successful_results = dna_results[dna_results['sharpe_ratio'].notna()]
            
            if len(successful_results) > 0:
                avg_profit = successful_results['net_profit_percentage'].mean()
                avg_sharpe = successful_results['sharpe_ratio'].mean()
                avg_balance = successful_results['finishing_balance'].mean()
                min_balance = successful_results['finishing_balance'].min()
            else:
                avg_profit = avg_sharpe = avg_balance = min_balance = 0
            
            # Normalize metrics for scoring (0-100 scale)
            # Success rate is already 0-1, multiply by 100
            norm_success = success_rate * 100
            
            # Profit: cap at 1000% and normalize
            norm_profit = min(avg_profit / 10, 100) if avg_profit > 0 else 0
            
            # Sharpe: assume 5.0 is excellent, normalize
            norm_sharpe = min(avg_sharpe / 5 * 100, 100) if avg_sharpe > 0 else 0
            
            # Balance: assume $500k is excellent
            norm_avg_balance = min(avg_balance / 500000 * 100, 100) if avg_balance > 0 else 0
            
            # Min balance: $50k+ gets full score, scale down from there
            norm_min_balance = min(min_balance / 50000 * 100, 100) if min_balance > 0 else 0
            
            # Use custom weights if provided, otherwise defaults
            if weights is None:
                weights = {
                    'success_rate': 0.50,  # CRITICAL: Avoid bankruptcy
                    'avg_profit': 0.20,
                    'avg_sharpe': 0.15,
                    'avg_balance': 0.10,
                    'min_balance': 0.05
                }
            
            # Calculate weighted composite score
            composite_score = (
                norm_success * weights.get('success_rate', 0.30) +
                norm_profit * weights.get('avg_profit', 0.25) +
                norm_sharpe * weights.get('avg_sharpe', 0.20) +
                norm_avg_balance * weights.get('avg_balance', 0.15) +
                norm_min_balance * weights.get('min_balance', 0.10)
            )
            
            dna_stats.append({
                'dna': dna_str,
                'success_rate': success_rate,
                'avg_profit': avg_profit,
                'avg_sharpe': avg_sharpe,
                'avg_balance': avg_balance,
                'min_balance': min_balance,
                'composite_score': composite_score
            })
        
        # Sort by composite score
        dna_stats_df = pd.DataFrame(dna_stats).sort_values('composite_score', ascending=False)
        
        # Award wins to top performers
        win_threshold = 40.0  # Minimum score to be considered a "win"
        min_success_rate = 0.80  # Minimum 80% success rate to avoid bankruptcy risk
        
        # Filter by both score AND success rate
        winners = dna_stats_df[
            (dna_stats_df['composite_score'] >= win_threshold) &
            (dna_stats_df['success_rate'] >= min_success_rate)
        ]
        
        # Alternative: Award top performer if they meet minimum criteria
        if len(winners) == 0 and len(dna_stats_df) > 0:
            # Only award if they have decent score AND high success rate
            top_dna = dna_stats_df.iloc[0]
            if top_dna['composite_score'] >= 30 and top_dna['success_rate'] >= min_success_rate:
                winners = dna_stats_df.head(1)
        
        stats = {
            'total_dnas_tested': len(dna_stats),
            'winners': len(winners),
            'winner_details': []
        }
        
        for idx, winner in winners.iterrows():
            # Get DNA hash
            dna_hash = hashlib.sha256(winner['dna'].encode()).hexdigest()[:16]
            
            # Update wins in database
            cursor.execute("""
                UPDATE dna_records 
                SET wins = wins + 1,
                    win_score = win_score + ?,
                    last_win_date = CURRENT_TIMESTAMP
                WHERE dna_hash = ?
            """, (winner['composite_score'], dna_hash))
            
            stats['winner_details'].append({
                'dna_hash': dna_hash,
                'composite_score': winner['composite_score'],
                'metrics': {
                    'success_rate': winner['success_rate'],
                    'avg_profit': winner['avg_profit'],
                    'avg_sharpe': winner['avg_sharpe'],
                    'avg_balance': winner['avg_balance'],
                    'min_balance': winner['min_balance']
                }
            })
        
        conn.commit()
        
        # Print summary
        print(f"\n🏆 WIN AWARDS")
        print("="*60)
        print(f"DNAs Tested: {stats['total_dnas_tested']}")
        print(f"Winners (score >= {win_threshold} AND success >= {min_success_rate:.0%}): {stats['winners']}")
        print("\nScoring Breakdown:")
        print("  - 50% Success Rate (avoid bankruptcy)")
        print("  - 20% Average Profit")
        print("  - 15% Average Sharpe")
        print("  - 10% Average Balance")
        print("  - 5% Minimum Balance")
        
        if winners.empty:
            print("\nNo DNAs achieved winning performance in this test batch.")
            if len(dna_stats_df) > 0:
                print("\nAll DNA Scores:")
                for _, dna in dna_stats_df.iterrows():
                    dna_hash = hashlib.sha256(dna['dna'].encode()).hexdigest()[:16]
                    print(f"  ...{dna_hash[-8:]} - Score: {dna['composite_score']:.1f}")
        else:
            print("\nTop Performers:")
            for _, winner in winners.head(3).iterrows():
                dna_hash = hashlib.sha256(winner['dna'].encode()).hexdigest()[:16]
                print(f"\n  ...{dna_hash[-8:]} - Score: {winner['composite_score']:.1f}")
                print(f"    Success: {winner['success_rate']:.0%}, Profit: {winner['avg_profit']:.0f}%")
                print(f"    Sharpe: {winner['avg_sharpe']:.2f}, Avg Balance: ${winner['avg_balance']:,.0f}")
        
        conn.close()
        return stats
    
    def get_leaderboard(self, min_wins: int = 1) -> pd.DataFrame:
        """
        Get DNA leaderboard sorted by wins and total win score
        
        Args:
            min_wins: Minimum number of wins to include
            
        Returns:
            DataFrame with DNA leaderboard
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                dna_hash,
                strategy_name,
                wins,
                win_score,
                win_score / NULLIF(wins, 0) as avg_win_score,
                last_win_date,
                sharpe_ratio as original_sharpe,
                net_profit_percentage as original_profit,
                symbol,
                timeframe
            FROM dna_records
            WHERE wins >= ?
            ORDER BY wins DESC, win_score DESC
        """
        
        df = pd.read_sql_query(query, conn, params=[min_wins])
        conn.close()
        
        return df


def get_hall_of_fame(db_path: Optional[str] = None) -> HallOfFame:
    """Get or create a Hall of Fame instance"""
    return HallOfFame(db_path)