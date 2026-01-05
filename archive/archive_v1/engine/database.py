"""Experiment database with SQLite persistence."""

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import time


@dataclass
class Trial:
    """A single experiment trial."""
    trial_id: str
    algorithm: str  # "eqprop" or "bp"
    task: str
    config: Dict[str, Any]
    seed: int
    
    # Results (populated after completion)
    performance: float = 0.0
    wall_time_seconds: float = 0.0
    memory_mb: float = 0.0
    
    # Metadata
    status: str = "pending"  # pending, running, complete, failed
    created_at: str = ""
    completed_at: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class Insight:
    """An insight generated from trial results."""
    insight_id: str
    insight_type: str  # "finding", "hypothesis_test", "comparison", "warning"
    title: str
    description: str
    evidence: List[str]  # List of trial IDs
    confidence: float  # 0-1
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ExperimentDB:
    """SQLite database for experiment persistence.
    
    Stores trials, insights, and campaign state.
    """
    
    def __init__(self, db_path: str = "results/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                task TEXT NOT NULL,
                config TEXT NOT NULL,
                seed INTEGER NOT NULL,
                performance REAL DEFAULT 0.0,
                wall_time_seconds REAL DEFAULT 0.0,
                memory_mb REAL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                created_at TEXT,
                completed_at TEXT,
                error_message TEXT
            )
        """)
        
        # Insights table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                insight_id TEXT PRIMARY KEY,
                insight_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                evidence TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT
            )
        """)
        
        # Campaign state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaign_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_trial(self, trial: Trial):
        """Add or update a trial."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO trials 
            (trial_id, algorithm, task, config, seed, performance, 
             wall_time_seconds, memory_mb, status, created_at, completed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial.trial_id, trial.algorithm, trial.task,
            json.dumps(trial.config), trial.seed, trial.performance,
            trial.wall_time_seconds, trial.memory_mb, trial.status,
            trial.created_at, trial.completed_at, trial.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get a trial by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_trial(row)
        return None
    
    def get_trials(self, algorithm: str = None, task: str = None, 
                   status: str = None) -> List[Trial]:
        """Get trials with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trials WHERE 1=1"
        params = []
        
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if task:
            query += " AND task = ?"
            params.append(task)
        if status:
            query += " AND status = ?"
            params.append(status)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_trial(row) for row in rows]
    
    def _row_to_trial(self, row) -> Trial:
        """Convert database row to Trial object."""
        return Trial(
            trial_id=row[0],
            algorithm=row[1],
            task=row[2],
            config=json.loads(row[3]),
            seed=row[4],
            performance=row[5],
            wall_time_seconds=row[6],
            memory_mb=row[7],
            status=row[8],
            created_at=row[9],
            completed_at=row[10],
            error_message=row[11] or ""
        )
    
    def add_insight(self, insight: Insight):
        """Add an insight."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO insights
            (insight_id, insight_type, title, description, evidence, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            insight.insight_id, insight.insight_type, insight.title,
            insight.description, json.dumps(insight.evidence),
            insight.confidence, insight.created_at
        ))
        
        conn.commit()
        conn.close()
    
    def get_insights(self, insight_type: str = None) -> List[Insight]:
        """Get all insights, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if insight_type:
            cursor.execute("SELECT * FROM insights WHERE insight_type = ?", (insight_type,))
        else:
            cursor.execute("SELECT * FROM insights")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Insight(
            insight_id=row[0],
            insight_type=row[1],
            title=row[2],
            description=row[3],
            evidence=json.loads(row[4]) if row[4] else [],
            confidence=row[5],
            created_at=row[6]
        ) for row in rows]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) FROM trials GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())
        
        # Count by task
        cursor.execute("""
            SELECT task, COUNT(*) FROM trials WHERE status = 'complete' GROUP BY task
        """)
        task_counts = dict(cursor.fetchall())
        
        # Count by algorithm
        cursor.execute("""
            SELECT algorithm, COUNT(*) FROM trials WHERE status = 'complete' GROUP BY algorithm
        """)
        algo_counts = dict(cursor.fetchall())
        
        # Best results per task
        cursor.execute("""
            SELECT task, algorithm, MAX(performance) 
            FROM trials WHERE status = 'complete' 
            GROUP BY task, algorithm
        """)
        best_results = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_trials": sum(status_counts.values()),
            "status_counts": status_counts,
            "task_counts": task_counts,
            "algorithm_counts": algo_counts,
            "best_results": [
                {"task": r[0], "algorithm": r[1], "performance": r[2]}
                for r in best_results
            ],
            "insights_count": len(self.get_insights()),
        }
    
    def export_json(self, path: str = None) -> str:
        """Export all data to JSON."""
        data = {
            "trials": [asdict(t) for t in self.get_trials()],
            "insights": [asdict(i) for i in self.get_insights()],
            "summary": self.get_summary(),
            "exported_at": datetime.now().isoformat(),
        }
        
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        
        return json.dumps(data, indent=2)
