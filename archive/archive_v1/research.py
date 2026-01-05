#!/usr/bin/env python3
"""
TorEqProp Unified Research System

A completely transparent autonomous research system that:
- Shows all parameters in real-time via live dashboard
- Runs fast preliminary experiments (~10s each)
- Collects results for final analysis and reports
- Provides parameter sensitivity analysis and heatmaps
- Guides the search towards promising results

Usage:
    python research.py                    # Full autonomous run (5 min default)
    python research.py --quick            # 2-minute rapid exploration
    python research.py --minutes 10       # Run for 10 minutes
    python research.py --no-dashboard     # Run headless (no TUI)
    python research.py --report           # Generate report from existing data
    python research.py --analyze          # Run parameter analysis only
"""

import argparse
import sys
import time
import random
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from research_engine import (
    ResearchConfig,
    ResultCollector,
    TimeAwareRunner,
    ProgressiveScheduler,
    ParameterAnalyzer,
    ResearchReporter,
    ResearchDashboard,
    TIERS,
    TIER_ORDER,
)
from research_engine.dashboard import create_dashboard, HAS_RICH


def run_research(
    minutes: float = 5.0,
    use_dashboard: bool = True,
    quick_mode: bool = False,
    output_dir: Path = Path("research_output"),
    max_experiment_time: float = 15.0,
    target_time_per_exp: float = 0.25,
):
    """
    Run the autonomous research loop.
    
    Args:
        minutes: Maximum runtime in minutes
        use_dashboard: Whether to show live TUI
        quick_mode: Use even smaller models for speed
        output_dir: Directory for results
        max_experiment_time: Max seconds per experiment
        target_time_per_exp: Target duration in minutes
    """
    # Configure
    config = ResearchConfig(
        output_dir=output_dir,
        max_experiment_time=max_experiment_time if not quick_mode else 5.0,
        total_budget_minutes=minutes,
        target_time_per_experiment=target_time_per_exp,
    )
    
    if quick_mode:
        config.d_model_values = [8, 16]
        config.eqprop_beta_values = [0.2, 0.25]
    
    # Initialize components
    collector = ResultCollector(config.output_dir)
    runner = TimeAwareRunner(collector, config)
    scheduler = ProgressiveScheduler(collector, config)
    analyzer = ParameterAnalyzer(collector, config)
    reporter = ResearchReporter(collector, config)
    
    # Set up time budget warning callback
    def time_warning(msg: str):
        if dashboard:
            dashboard.add_warning(msg)
        else:
            print(msg)
    
    runner.on_time_warning = time_warning
    
    # Create dashboard
    dashboard = None
    if use_dashboard and HAS_RICH:
        dashboard = create_dashboard(use_rich=True, collector=collector, config=config)
    elif use_dashboard:
        dashboard = create_dashboard(use_rich=False, collector=collector, config=config)
    
    deadline = time.time() + minutes * 60
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("🔬 TorEqProp Unified Research Engine")
    print(f"{'='*60}")
    print(f"⏱️  Budget: {minutes:.1f} minutes")
    print(f"📁 Output: {config.output_dir}")
    print(f"🎯 Mode: {'Quick' if quick_mode else 'Standard'}")
    print(f"{'='*60}\n")
    
    try:
        if dashboard:
            dashboard.start(deadline=deadline)
        
        experiment_count = 0
        
        while time.time() < deadline:
            # Get next experiment from scheduler
            task, algorithm, exp_config, tier = scheduler.get_next_experiment()
            
            # Update dashboard
            if dashboard:
                dashboard.update_experiment(task, algorithm, exp_config, "Starting...")
                dashboard.state.tier_progress = scheduler.get_progress()
                dashboard.update()
            
            # Get tier time limit
            tier_info = TIERS.get(tier)
            max_time = tier_info.max_time_per_trial if tier_info else config.max_experiment_time
            
            # Run experiment
            seed = random.randint(0, 9999)
            trial = runner.run(
                task=task,
                algorithm=algorithm,
                config=exp_config,
                seed=seed,
                tier=tier,
                max_time=max_time,
            )
            
            experiment_count += 1
            
            # Record result
            scheduler.mark_completed(task, algorithm, exp_config)
            
            if dashboard:
                dashboard.record_result(trial)
                
                # Check for discoveries
                if trial.status == "complete" and trial.performance > 0.8:
                    algo_name = "EqProp" if algorithm == "eqprop" else "BP"
                    dashboard.add_discovery(f"{algo_name} achieves {trial.performance:.1%} on {task}")
                
                dashboard.update()
            else:
                # Simple progress output
                status = "✅" if trial.status == "complete" else "❌"
                algo_icon = "🔋" if algorithm == "eqprop" else "⚡"
                print(f"{status} {algo_icon} {task}: {trial.performance:.4f} ({trial.cost.wall_time_seconds:.1f}s)")
            
            # Brief pause for responsiveness
            time.sleep(0.1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Research interrupted by user")
    
    finally:
        if dashboard:
            dashboard.stop()
            dashboard.print_summary()
    
    # Generate report
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"📊 Generating final report...")
    
    report_path = reporter.generate_final_report(analyzer)
    print(f"📄 Report saved: {report_path}")
    
    # Generate heatmaps if we have enough data
    trials = collector.get_trials(status="complete")
    if len(trials) >= 10:
        print("📈 Generating parameter heatmaps...")
        heatmaps = analyzer.generate_all_heatmaps()
        for hm in heatmaps:
            print(f"   📊 {hm}")
    
    # Save analysis JSON
    analysis_path = analyzer.save_analysis_json()
    print(f"📋 Analysis data: {analysis_path}")
    
    print(f"\n✅ Research complete!")
    print(f"   ⏱️  Time: {elapsed/60:.1f} minutes")
    print(f"   🧪 Experiments: {experiment_count}")
    print(f"   📁 Results: {config.output_dir}")
    print(f"{'='*60}\n")
    
    return report_path


def generate_report_only(output_dir: Path = Path("research_output")):
    """Generate report from existing data."""
    config = ResearchConfig(output_dir=output_dir)
    collector = ResultCollector(config.output_dir)
    analyzer = ParameterAnalyzer(collector, config)
    reporter = ResearchReporter(collector, config)
    
    trials = collector.get_trials(status="complete")
    
    if not trials:
        print("❌ No completed trials found. Run experiments first.")
        return None
    
    print(f"📊 Found {len(trials)} completed trials")
    print("📝 Generating report...")
    
    report_path = reporter.generate_final_report(analyzer)
    print(f"📄 Report saved: {report_path}")
    
    # Parameter analysis
    print("\n📈 Parameter Importance:")
    importance = analyzer.sensitivity_analysis(trials)
    for param, f_stat in list(importance.items())[:5]:
        print(f"   {param}: F={f_stat:.2f}")
    
    return report_path


def run_analysis_only(output_dir: Path = Path("research_output")):
    """Run parameter analysis on existing data."""
    config = ResearchConfig(output_dir=output_dir)
    collector = ResultCollector(config.output_dir)
    analyzer = ParameterAnalyzer(collector, config)
    
    trials = collector.get_trials(status="complete")
    
    if len(trials) < 5:
        print(f"❌ Need at least 5 trials for analysis (found {len(trials)})")
        return
    
    print(f"📊 Analyzing {len(trials)} trials...")
    
    # Importance analysis
    print("\n📈 Parameter Importance (ANOVA F-statistics):")
    importance = analyzer.sensitivity_analysis(trials)
    for param, f_stat in importance.items():
        bars = "█" * int(min(f_stat, 20))
        print(f"   {param:15s}: {f_stat:8.2f} {bars}")
    
    # Best values
    print("\n🏆 Best Values per Parameter:")
    best_values = analyzer.get_best_value_per_param(trials)
    for param, (value, perf) in best_values.items():
        val_str = f"{value:.4g}" if isinstance(value, float) else str(value)
        print(f"   {param:15s}: {val_str:10s} → {perf:.4f}")
    
    # Generate heatmaps
    print("\n📊 Generating heatmaps...")
    heatmaps = analyzer.generate_all_heatmaps()
    for hm in heatmaps:
        print(f"   ✅ {hm}")
    
    # Save JSON
    json_path = analyzer.save_analysis_json()
    print(f"\n📋 Full analysis saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Unified Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python research.py                    # Run for 5 minutes with dashboard
    python research.py --quick            # Quick 2-minute exploration
    python research.py --minutes 10       # Run for 10 minutes
    python research.py --no-dashboard     # Run without TUI
    python research.py --report           # Generate report from existing data
    python research.py --analyze          # Analyze existing results
        """
    )
    
    parser.add_argument(
        "--minutes", "-m",
        type=float,
        default=5.0,
        help="Maximum runtime in minutes (default: 5)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: 2 minutes with smaller models"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run without live TUI dashboard"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate report from existing data only"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run parameter analysis on existing data"
    )
    parser.add_argument(
        "--max-time", 
        type=float,
        default=15.0,
        help="Max seconds per experiment (default: 15.0)"
    )
    parser.add_argument(
        "--target-time", 
        type=float,
        default=0.25,
        help="Target minutes per experiment (default: 0.25)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("research_output"),
        help="Output directory (default: research_output)"
    )
    
    args = parser.parse_args()
    
    if args.report:
        generate_report_only(args.output)
    elif args.analyze:
        run_analysis_only(args.output)
    else:
        minutes = 2.0 if args.quick else args.minutes
        run_research(
            minutes=minutes,
            use_dashboard=not args.no_dashboard,
            quick_mode=args.quick,
            output_dir=args.output,
            max_experiment_time=args.max_time,
            target_time_per_exp=args.target_time,
        )


if __name__ == "__main__":
    main()
