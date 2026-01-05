import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from .core import HyperOptTrial

class HyperOptDB:
    """Database for storing hyperopt trials."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.trials: Dict[str, HyperOptTrial] = {}
        self._load()
    
    def _load(self):
        """Load trials from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    content = f.read().strip()
                    if not content:
                        return  # Empty file
                    data = json.loads(content)
                    for trial_data in data.get("trials", []):
                        trial = HyperOptTrial.from_dict(trial_data)
                        self.trials[trial.trial_id] = trial
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load DB from {self.db_path}: {e}")
                self.trials = {}
    
    def _save(self):
        """Save trials to disk."""
        data = {
            "trials": [t.to_dict() for t in self.trials.values()],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_trial(self, trial: HyperOptTrial):
        """Add or update a trial."""
        self.trials[trial.trial_id] = trial
        self._save()
    
    def get_trial(self, trial_id: str) -> Optional[HyperOptTrial]:
        """Get a trial by ID."""
        return self.trials.get(trial_id)
    
    def get_trials(self, algorithm: str = None, task: str = None,
                   status: str = None) -> List[HyperOptTrial]:
        """Get trials matching filters."""
        result = list(self.trials.values())
        
        if algorithm:
            result = [t for t in result if t.algorithm == algorithm]
        if task:
            result = [t for t in result if t.task == task]
        if status:
            result = [t for t in result if t.status == status]
        
        return result
    
    def get_best_trial(self, algorithm: str, task: str) -> Optional[HyperOptTrial]:
        """Get best-performing trial for algorithm and task."""
        trials = self.get_trials(algorithm=algorithm, task=task, status="complete")
        if not trials:
            return None
        return max(trials, key=lambda t: t.performance)
