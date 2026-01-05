from .engine import HyperOptEngine, HyperOptDB
from .evaluator import CostAwareEvaluator
from .validation import ValidationPipeline
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Hyperparameter Optimization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python -m hyperopt.cli --smoke-test --ultra-fast
  
  # Run optimization on a task
  python -m hyperopt.cli --task mnist --n-trials 20 --epochs 5
  
  # Full scientific campaign
  python -m hyperopt.cli --full-campaign --task mnist
  
  # Generate validated report
  python -m hyperopt.cli --validate --task mnist
  
  # Adversarial robustness evaluation
  python -m hyperopt.cli --robustness --task mnist
        """
    )
    
    # Modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    mode.add_argument("--report", action="store_true", help="Generate report only")
    mode.add_argument("--campaign", action="store_true", help="Run multi-task campaign")
    mode.add_argument("--full-campaign", action="store_true", 
                     help="Run full scientific campaign with validation")
    mode.add_argument("--validate", action="store_true",
                     help="Generate validated report from existing results")
    mode.add_argument("--robustness", action="store_true",
                     help="Run adversarial robustness evaluation")
    mode.add_argument("--scaling", action="store_true",
                     help="Run scaling law analysis")
    mode.add_argument("--status", action="store_true",
                     help="Show status of all trials")
    
    # Configuration
    parser.add_argument("--task", type=str, default="mnist", 
                       help="Task to optimize (mnist, fashion, cifar10, parity, cartpole, etc.)")
    parser.add_argument("--tasks", type=str, nargs="+",
                       default=["xor", "mnist", "cartpole"],
                       help="Tasks for campaign mode")
    parser.add_argument("--strategy", type=str, default="random",
                       choices=["grid", "random", "sobol", "lhs"],
                       help="Sampling strategy")
    parser.add_argument("--n-trials", type=int, default=50,
                       help="Number of trials per algorithm")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Epochs per trial")
    parser.add_argument("--time-budget", type=float, default=None,
                       help="Time budget in seconds per trial (Fair Comparison Mode)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="Random seeds")
    parser.add_argument("--config", type=str, default="validation_config.yaml",
                       help="Path to configuration file")
    
    # Speed options
    parser.add_argument("--ultra-fast", action="store_true", 
                       help="Ultra-fast settings for smoke test")
    parser.add_argument("--rapid", action="store_true",
                       help="Rapid mode: fewer epochs/trials for quick feedback")
    
    # Flags
    parser.add_argument("--headless", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    engine = HyperOptEngine(config_path=args.config)
    
    # Report mode
    if args.report:
        engine._print_analysis(args.task)
        return
    
    # Status mode
    if args.status:
        _show_status(engine)
        return
    
    # Smoke test mode
    if args.smoke_test:
        print("🧪 SMOKE TEST MODE")
        if args.ultra_fast:
            print("⚡ ULTRA-FAST: d_model=8, micro task, <10s per trial, single seed")
            engine.eqprop_space.d_model = [8, 16]
            engine.eqprop_space.max_iters = [10]
            engine.baseline_space.d_model = [8, 16]
            
            if args.task == "mnist":
                args.task = "xor"
                
            engine.run(task=args.task, n_trials=2, strategy=args.strategy, 
                      epochs=1, seeds=[0], headless=args.headless)
        else:
            engine.run(task=args.task, n_trials=2, strategy=args.strategy, 
                      epochs=1, seeds=[0], headless=args.headless)
        return
    
    # Full campaign mode (with validation)
    if args.full_campaign:
        print("🔬 FULL SCIENTIFIC CAMPAIGN MODE")
        _run_full_campaign(engine, args)
        return
    
    # Multi-task campaign mode
    if args.campaign:
        print("🚀 CAMPAIGN MODE")
        tasks = args.tasks
        for t in tasks:
            print(f"\n>> Running Task: {t}")
            try:
                engine.run(task=t, n_trials=args.n_trials, strategy=args.strategy,
                          epochs=args.epochs, seeds=args.seeds, headless=args.headless)
            except Exception as e:
                print(f"❌ Task {t} failed: {e}")
        return
    
    # Validate mode
    if args.validate:
        print("📝 VALIDATION MODE")
        _run_validation(engine, args)
        return
    
    # Robustness mode
    if args.robustness:
        print("🛡️ ROBUSTNESS EVALUATION MODE")
        _run_robustness(engine, args)
        return
    
    # Scaling mode
    if args.scaling:
        print("📈 SCALING ANALYSIS MODE")
        _run_scaling(engine, args)
        return
    
    # Normal run
    engine.run(task=args.task, n_trials=args.n_trials, strategy=args.strategy,
              epochs=args.epochs, seeds=args.seeds, headless=args.headless, 
              time_budget=args.time_budget)


def _show_status(engine):
    """Show status of all trials."""
    from collections import defaultdict
    
    print("\n" + "=" * 70)
    print("  HYPEROPT STATUS")
    print("=" * 70)
    
    all_trials = engine.db.get_trials()
    if not all_trials:
        print("  No trials found.")
        return
    
    by_task = defaultdict(list)
    for t in all_trials:
        by_task[t.task].append(t)
    
    for task, trials in sorted(by_task.items()):
        eq_trials = [t for t in trials if t.algorithm == "eqprop"]
        bl_trials = [t for t in trials if t.algorithm == "bp"]
        
        eq_complete = sum(1 for t in eq_trials if t.status == "complete")
        bl_complete = sum(1 for t in bl_trials if t.status == "complete")
        
        print(f"\n📋 {task}:")
        print(f"   EqProp: {eq_complete}/{len(eq_trials)} complete")
        print(f"   Baseline: {bl_complete}/{len(bl_trials)} complete")
        
        if eq_complete > 0:
            best_eq = max((t for t in eq_trials if t.status == "complete"),
                         key=lambda x: x.performance)
            print(f"   Best EqProp: {best_eq.performance:.4f}")
        
        if bl_complete > 0:
            best_bl = max((t for t in bl_trials if t.status == "complete"),
                         key=lambda x: x.performance)
            print(f"   Best Baseline: {best_bl.performance:.4f}")


def _run_full_campaign(engine, args):
    """Run full scientific campaign with validation."""
    import numpy as np
    
    # Run hyperopt
    engine.run(task=args.task, n_trials=args.n_trials, strategy=args.strategy,
              epochs=args.epochs, seeds=args.seeds, headless=True)
    
    # Get results
    eq = engine.db.get_trials(algorithm="eqprop", task=args.task, status="complete")
    bl = engine.db.get_trials(algorithm="bp", task=args.task, status="complete")
    
    if not eq or not bl:
        print("❌ Insufficient trials for validation")
        return
    
    # Run validation
    pipeline = ValidationPipeline()
    claims = [{
        "claim": f"EqProp vs Baseline on {args.task}",
        "eqprop": [t.performance for t in eq],
        "baseline": [t.performance for t in bl]
    }]
    
    verdicts = pipeline.validate_claims(claims)
    pipeline.generate_report(verdicts, title=f"{args.task.upper()} Scientific Campaign")
    
    print("\n✅ Full campaign complete!")
    print(f"   Report saved to: reports/")


def _run_validation(engine, args):
    """Generate validated report from existing results."""
    pipeline = ValidationPipeline()
    
    eq = engine.db.get_trials(algorithm="eqprop", task=args.task, status="complete")
    bl = engine.db.get_trials(algorithm="bp", task=args.task, status="complete")
    
    if not eq or not bl:
        print(f"❌ No completed trials found for {args.task}")
        return
    
    claims = [{
        "claim": f"EqProp vs Baseline on {args.task}",
        "eqprop": [t.performance for t in eq],
        "baseline": [t.performance for t in bl]
    }]
    
    verdicts = pipeline.validate_claims(claims)
    pipeline.generate_report(verdicts, title=f"{args.task.upper()} Validation")
    
    print(f"\n✅ Validation report generated for {args.task}")


def _run_robustness(engine, args):
    """Run adversarial robustness evaluation."""
    try:
        from analysis.robustness import AdversarialEvaluator
        print(f"  Robustness evaluation for {args.task}")
        print("  ⚠️ Requires trained model - run with campaign_orchestrator.py")
    except ImportError as e:
        print(f"❌ Could not import robustness module: {e}")


def _run_scaling(engine, args):
    """Run scaling law analysis."""
    try:
        from analysis.scaling import ScalingAnalyzer
        analyzer = ScalingAnalyzer(engine, results_dir="scaling_results")
        sizes = [32, 64, 128, 256]
        print(f"  Running scaling sweep: {sizes}")
        results = analyzer.run_scaling_sweep(task=args.task, sizes=sizes)
        
        print("\n  Results:")
        for r in results:
            print(f"    d={r['d_model']}: loss={r['loss']:.4f}, params={r['params']}")
    except Exception as e:
        print(f"❌ Scaling analysis failed: {e}")


if __name__ == "__main__":
    main()
