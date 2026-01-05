"""
CLI for TEP experiments.

Usage:
    # Quick smoke test
    python -m tep_experiment --smoke-test
    
    # Phase 1: Rapid Signal Detection (6-10 hours)
    python -m tep_experiment --phase 1
    
    # Full pipeline with all phases
    python -m tep_experiment --full
    
    # Custom configuration
    python -m tep_experiment --phase 1 --n-trials 100 --budget-hours 4
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="TEP Experiment: Rigorous TEP vs BP Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation that everything works
    python -m tep_experiment --smoke-test
    
    # Run Phase 1 with default settings (6-10 hours)
    python -m tep_experiment --phase 1
    
    # Run Phase 1 with reduced budget for testing
    python -m tep_experiment --phase 1 --n-trials 50 --budget-hours 2
    
    # Run full pipeline (all phases)
    python -m tep_experiment --full
    
    # Launch Optuna dashboard for monitoring
    python -m tep_experiment --dashboard
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test (~5 minutes)"
    )
    mode_group.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Run a specific phase"
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline (all phases with gating)"
    )
    mode_group.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch Optuna dashboard for monitoring"
    )
    
    # Configuration
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of trials per algorithm (overrides phase default)"
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="Total budget in hours (overrides phase default)"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///tep_experiments.db",
        help="Optuna storage URL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tep_results",
        help="Output directory for results"
    )
    
    # Advanced
    parser.add_argument(
        "--no-seed-transfer",
        action="store_true",
        help="Disable transfer seeding from previous phases"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle dashboard mode separately
    if args.dashboard:
        launch_dashboard(args.storage)
        return
    
    # Import engine (lazy to avoid slow import on --help)
    from .engine import TEPExperimentEngine
    from .config import PHASE_CONFIGS, SMOKE_TEST_CONFIG
    
    # Create engine
    engine = TEPExperimentEngine(
        storage_url=args.storage,
        output_dir=Path(args.output_dir),
    )
    
    # Apply overrides if provided
    if args.n_trials or args.budget_hours:
        if args.phase:
            config = PHASE_CONFIGS[args.phase]
            if args.n_trials:
                config.n_trials_per_algorithm = args.n_trials
            if args.budget_hours:
                config.total_budget_hours = args.budget_hours
    
    # Run appropriate mode
    if args.smoke_test:
        results = engine.run_smoke_test()
        print("\n📋 Smoke Test Results:")
        for task, data in results.items():
            print(f"   {task}: {data['comparison']['winner']}")
    
    elif args.phase:
        print(f"\n🚀 Running Phase {args.phase}")
        results = engine.run_phase(
            phase=args.phase,
            seed_from_previous=not args.no_seed_transfer,
        )
        print(f"\nPhase {args.phase} complete!")
        print(f"Success: {'✅' if results['success'] else '❌'}")
    
    elif args.full:
        print("\n🔬 Running Full Pipeline")
        results = engine.run_full_pipeline()
        print(f"\nPipeline complete!")
        print(f"Phase reached: {results['phase_reached']}")
        print(f"Report: {results['final_report_path']}")


def launch_dashboard(storage_url: str):
    """Launch Optuna dashboard for experiment monitoring."""
    try:
        import optuna_dashboard
        print(f"🖥️  Launching Optuna Dashboard...")
        print(f"   Storage: {storage_url}")
        print(f"   Open http://localhost:8080 in your browser")
        print(f"   Press Ctrl+C to stop")
        
        optuna_dashboard.run_server(storage_url, host="0.0.0.0", port=8080)
        
    except ImportError:
        print("❌ optuna-dashboard not installed.")
        print("   Install with: pip install optuna-dashboard")
        print("   Or use: optuna-dashboard sqlite:///tep_experiments.db")
        sys.exit(1)


if __name__ == "__main__":
    main()
