#!/usr/bin/env python3
"""
Generate insights from existing TorEqProp research data.

Answers the key questions:
- When could TorEqProp outperform BP?
- What hyperparameters are most sensitive?
- Where should we invest more compute?
"""

import json
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_hyperopt_data():
    """Analyze data/hyperopt_results.json"""
    data_file = Path("data/hyperopt_results.json")
    if not data_file.exists():
        return None
    
    with open(data_file) as f:
        data = json.load(f)
    
    trials = data.get("trials", [])
    if not trials:
        return None
    
    print(f"\n{'='*60}")
    print(f"📊 HYPEROPT DATA ANALYSIS ({len(trials)} trials)")
    print(f"{'='*60}\n")
    
    # Group by algorithm and task
    by_algo_task = defaultdict(list)
    for trial in trials:
        if trial.get("status") == "complete" and trial.get("performance", 0) > 0:
            key = (trial.get("algorithm"), trial.get("task"))
            by_algo_task[key].append(trial)
    
    # Q1: When does EqProp outperform BP?
    print("❓ Q1: When does EqProp outperform BP?\n")
    
    eqprop_wins = []
    bp_wins = []
    
    for (algo, task), task_trials in by_algo_task.items():
        if not task_trials:
            continue
        best = max(task_trials, key=lambda t: t.get("performance", 0))
        
        if algo == "eqprop":
            eqprop_best = best["performance"]
            bp_best = max([t["performance"] for t in by_algo_task.get(("bp", task), [])], default=0)
            
            if eqprop_best > bp_best and bp_best > 0:
                diff = (eqprop_best - bp_best) / bp_best * 100
                print(f"   ✅ {task}: EqProp {eqprop_best:.3f} vs BP {bp_best:.3f} (+{diff:.1f}%)")
                print(f"      Config: β={best['config'].get('beta')}, d_model={best['config'].get('d_model')}, damping={best['config'].get('damping')}")
                eqprop_wins.append((task, best['config']))
    
    if not eqprop_wins:
        print("   ⚠️ No tasks where EqProp outperforms BP in current data\n")
    
    # Q2: What hyperparameters are most sensitive?
    print("\n❓ Q2: What hyperparameters are most sensitive?\n")
    
    eq_trials = [t for t in trials if t.get("algorithm") == "eqprop" and t.get("status") == "complete"]
    
    if len(eq_trials) >= 3:
        # Analyze beta sensitivity
        beta_groups = defaultdict(list)
        for trial in eq_trials:
            beta = trial.get("config", {}).get("beta")
            if beta and trial.get("performance", 0) > 0:
                beta_groups[beta].append(trial["performance"])
        
        if len(beta_groups) >= 2:
            print("   📊 Beta sensitivity:")
            for beta in sorted(beta_groups.keys()):
                perfs = beta_groups[beta]
                print(f"      β={beta}: avg={np.mean(perfs):.3f}, std={np.std(perfs):.3f}, n={len(perfs)}")
        
        # Analyze d_model sensitivity
        dmodel_groups = defaultdict(list)
        for trial in eq_trials:
            d = trial.get("config", {}).get("d_model")
            if d and trial.get("performance", 0) > 0:
                dmodel_groups[d].append(trial["performance"])
        
        if len(dmodel_groups) >= 2:
            print("\n   📊 d_model sensitivity:")
            for d in sorted(dmodel_groups.keys()):
                perfs = dmodel_groups[d]
                print(f"      d_model={d}: avg={np.mean(perfs):.3f}, std={np.std(perfs):.3f}, n={len(perfs)}")
    
    # Q3: Where to invest more compute?
    print("\n❓ Q3: Where should we invest more compute?\n")
    
    # Find promising configs with few samples
    config_performance = defaultdict(list)
    for trial in eq_trials:
        config_str = f"β={trial.get('config', {}).get('beta')}, d={trial.get('config', {}).get('d_model')}"
        if trial.get("performance", 0) > 0:
            config_performance[config_str].append(trial["performance"])
    
    promising = []
    for config_str, perfs in config_performance.items():
        if len(perfs) >= 2 and np.mean(perfs) > 0.7:
            promising.append((config_str, np.mean(perfs), len(perfs)))
    
    if promising:
        print("   🎯 Promising configurations (need more seeds):")
        for config, avg, n in sorted(promising, key=lambda x: -x[1])[:5]:
            print(f"      {config}: avg={avg:.3f}, n={n} {'✓ validated' if n >= 3 else '⚠️ needs more runs'}")
    
    return trials


def analyze_campaign_db():
    """Analyze results/test_campaign.db"""
    db_path = Path("results/test_campaign.db")
    if not db_path.exists():
        return None
    
    print(f"\n{'='*60}")
    print("📊 CAMPAIGN DATABASE ANALYSIS")
    print(f"{'='*60}\n")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get summary
    cursor.execute("""
        SELECT algorithm, status, COUNT(*), AVG(performance), MAX(performance)
        FROM trials
        GROUP BY algorithm, status
    """)
    
    print("Summary by algorithm/status:")
    for algo, status, count, avg_perf, max_perf in cursor.fetchall():
        print(f"   {algo} ({status}): {count} trials, avg={avg_perf or 0:.3f}, best={max_perf or 0:.3f}")
    
    # Best configs
    cursor.execute("""
        SELECT task, algorithm, performance, config
        FROM trials
        WHERE status = 'complete' AND performance > 0
        ORDER BY performance DESC
        LIMIT 10
    """)
    
    print("\nTop 10 configurations:")
    for i, (task, algo, perf, config_json) in enumerate(cursor.fetchall(), 1):
        config = json.loads(config_json) if config_json else {}
        beta = config.get('beta', 'N/A')
        d_model = config.get('d_model', 'N/A')
        print(f"   {i}. {algo}/{task}: {perf:.4f} (β={beta}, d={d_model})")
    
    conn.close()


def main():
    print("\n" + "="*60)
    print("🔬 TorEqProp Research Insights Generator")
    print("="*60)
    
    # Analyze all available data
    analyze_hyperopt_data()
    analyze_campaign_db()
    
    print(f"\n{'='*60}")
    print("📝 RECOMMENDATIONS")
    print(f"{'='*60}\n")
    print("1. Focus experiments on configurations where EqProp shows promise")
    print("2. Run more seeds for promising β values to validate")
    print("3. Test EqProp on tasks where equilibrium dynamics might help:")
    print("   - RL tasks (CartPole, MountainCar)")
    print("   - Sequence tasks requiring memory")
    print("4. Compare time-to-accuracy, not just final accuracy")
    print("\n")

if __name__ == "__main__":
    main()
