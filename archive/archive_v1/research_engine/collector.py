"""
Result collection and storage for the research engine.

Provides unified storage with both JSON (human-readable) and SQLite (queryable) backends.
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


@dataclass
class CostMetrics:
    """Cost metrics for a single trial."""
    wall_time_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    total_iterations: int = 0
    param_count: int = 0
    epochs_completed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class Trial:
    """A single experiment trial with full metadata."""
    trial_id: str
    algorithm: str  # "eqprop" or "bp"
    task: str
    config: Dict[str, Any]
    seed: int
    
    # Results
    performance: float = 0.0
    performance_metric: str = "accuracy"
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Status
    status: str = "pending"  # pending, running, complete, failed, timeout
    error: str = ""
    tier: str = "micro"
    
    # Timestamps
    started_at: str = ""
    completed_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "algorithm": self.algorithm,
            "task": self.task,
            "config": self.config,
            "seed": self.seed,
            "performance": self.performance,
            "performance_metric": self.performance_metric,
            "cost": self.cost.to_dict(),
            "status": self.status,
            "error": self.error,
            "tier": self.tier,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trial":
        cost_data = d.get("cost", {})
        if isinstance(cost_data, dict):
            cost = CostMetrics(**cost_data)
        else:
            cost = CostMetrics()
        
        return cls(
            trial_id=d["trial_id"],
            algorithm=d["algorithm"],
            task=d["task"],
            config=d.get("config", {}),
            seed=d.get("seed", 0),
            performance=d.get("performance", 0.0),
            performance_metric=d.get("performance_metric", "accuracy"),
            cost=cost,
            status=d.get("status", "complete"),
            error=d.get("error", ""),
            tier=d.get("tier", "micro"),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
        )


class ResultCollector:
    """Unified result storage with multiple backends."""
    
    def __init__(self, output_dir: Path = Path("research_output")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.output_dir / "results.json"
        self.sqlite_path = self.output_dir / "results.db"
        
        self._init_sqlite()
        self._init_json()
    
    def _init_sqlite(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id TEXT PRIMARY KEY,
                algorithm TEXT,
                task TEXT,
                config TEXT,
                seed INTEGER,
                performance REAL,
                performance_metric TEXT,
                wall_time REAL,
                param_count INTEGER,
                status TEXT,
                error TEXT,
                tier TEXT,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_algo ON trials(task, algorithm)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON trials(status)
        """)
        conn.commit()
        conn.close()
    
    def _init_json(self):
        """Initialize JSON file if not exists."""
        if not self.json_path.exists():
            self._write_json({"trials": [], "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0",
            }})
    
    def _read_json(self) -> Dict:
        """Read JSON database."""
        if self.json_path.exists():
            with open(self.json_path, "r") as f:
                return json.load(f)
        return {"trials": [], "metadata": {}}
    
    def _write_json(self, data: Dict):
        """Write JSON database."""
        with open(self.json_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def save_trial(self, trial: Trial):
        """Save trial to both JSON and SQLite."""
        # SQLite
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO trials 
            (trial_id, algorithm, task, config, seed, performance, performance_metric,
             wall_time, param_count, status, error, tier, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial.trial_id,
            trial.algorithm,
            trial.task,
            json.dumps(trial.config),
            trial.seed,
            trial.performance,
            trial.performance_metric,
            trial.cost.wall_time_seconds,
            trial.cost.param_count,
            trial.status,
            trial.error,
            trial.tier,
            trial.started_at,
            trial.completed_at,
        ))
        conn.commit()
        conn.close()
        
        # JSON (append)
        data = self._read_json()
        # Remove existing trial with same ID if present
        data["trials"] = [t for t in data["trials"] if t.get("trial_id") != trial.trial_id]
        data["trials"].append(trial.to_dict())
        data["metadata"]["updated"] = datetime.now().isoformat()
        self._write_json(data)
    
    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get a specific trial by ID."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_trial(row)
        return None
    
    def get_trials(
        self,
        task: Optional[str] = None,
        algorithm: Optional[str] = None,
        status: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Trial]:
        """Query trials with filters."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trials WHERE 1=1"
        params = []
        
        if task:
            query += " AND task = ?"
            params.append(task)
        if algorithm:
            query += " AND algorithm = ?"
            params.append(algorithm)
        if status:
            query += " AND status = ?"
            params.append(status)
        if tier:
            query += " AND tier = ?"
            params.append(tier)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_trial(row) for row in rows]
    
    def _row_to_trial(self, row) -> Trial:
        """Convert SQLite row to Trial object."""
        return Trial(
            trial_id=row[0],
            algorithm=row[1],
            task=row[2],
            config=json.loads(row[3]) if row[3] else {},
            seed=row[4],
            performance=row[5],
            performance_metric=row[6],
            cost=CostMetrics(
                wall_time_seconds=row[7] or 0.0,
                param_count=row[8] or 0,
            ),
            status=row[9],
            error=row[10] or "",
            tier=row[11] or "micro",
            started_at=row[12] or "",
            completed_at=row[13] or "",
        )
    
    def get_all_trials(self) -> List[Trial]:
        """Get all completed trials."""
        return self.get_trials(status="complete")
    
    def get_best_trial(self, task: str, algorithm: str) -> Optional[Trial]:
        """Get best performing trial for a task/algorithm."""
        trials = self.get_trials(task=task, algorithm=algorithm, status="complete")
        if not trials:
            return None
        return max(trials, key=lambda t: t.performance)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total counts
        cursor.execute("SELECT COUNT(*) FROM trials")
        stats["total_trials"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trials WHERE status = 'complete'")
        stats["completed_trials"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM trials WHERE status = 'failed' OR status = 'timeout'")
        stats["failed_trials"] = cursor.fetchone()[0]
        
        # Best per task/algorithm
        cursor.execute("""
            SELECT task, algorithm, MAX(performance) 
            FROM trials 
            WHERE status = 'complete'
            GROUP BY task, algorithm
        """)
        stats["best_by_task_algo"] = {
            f"{row[0]}_{row[1]}": row[2] for row in cursor.fetchall()
        }
        
        # Win counts
        cursor.execute("""
            SELECT task,
                   MAX(CASE WHEN algorithm = 'eqprop' THEN performance END) as eq_best,
                   MAX(CASE WHEN algorithm = 'bp' THEN performance END) as bp_best
            FROM trials
            WHERE status = 'complete'
            GROUP BY task
        """)
        eqprop_wins = 0
        bp_wins = 0
        ties = 0
        for row in cursor.fetchall():
            eq = row[1] or 0
            bp = row[2] or 0
            if eq > bp:
                eqprop_wins += 1
            elif bp > eq:
                bp_wins += 1
            else:
                ties += 1
        
        stats["eqprop_wins"] = eqprop_wins
        stats["bp_wins"] = bp_wins
        stats["ties"] = ties
        
        conn.close()
        return stats
    
    def export_for_analysis(self) -> List[Dict]:
        """Export all trials as list of dicts for analysis."""
        trials = self.get_all_trials()
        return [t.to_dict() for t in trials]
    
    def generate_checksum(self) -> str:
        """Generate SHA-256 checksum of results for integrity verification."""
        data = self._read_json()
        json_str = json.dumps(data["trials"], sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
