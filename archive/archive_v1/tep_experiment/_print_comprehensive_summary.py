    def _print_comprehensive_summary(
        self,
        task: str,
        tep_front: ParetoFront,
        bp_front: ParetoFront,
        comparison: Dict[str, Any],
        tep_trials: List[TrialResult],
        bp_trials: List[TrialResult],
    ):
        """Print comprehensive summary with derived metrics and top configs."""
        
        print(f"\n   📈 Results for {task}:")
        print(f"      TEP: {len(tep_front.points)} Pareto points")
        print(f"      BP:  {len(bp_front.points)} Pareto points")
        
        # Get best configs
        tep_best = tep_front.best_accuracy()
        bp_best = bp_front.best_accuracy()
        
        if tep_best and bp_best:
            # Compute derived metrics
            tep_derived = compute_derived_metrics(
                tep_best.accuracy,
                tep_best.wall_time,
                tep_best.param_count,
                tep_best.convergence_steps
            )
            bp_derived = compute_derived_metrics(
                bp_best.accuracy,
                bp_best.wall_time,
                bp_best.param_count,
                bp_best.convergence_steps
            )
            
            # Print base objectives
            print(f"\n   🎯 Best Configurations:")
            print(f"      TEP: acc={tep_best.accuracy:.4f}, time={tep_best.wall_time:.1f}s, params={tep_best.param_count}")
            print(f"      BP:  acc={bp_best.accuracy:.4f}, time={bp_best.wall_time:.1f}s, params={bp_best.param_count}")
            
            # Print derived metrics
            print(f"\n   ⚡ Efficiency Metrics:")
            print(f"      Learning Power (acc/time):")
            print(f"        TEP: {tep_derived.learning_power:.4f} acc/s")
            print(f"        BP:  {bp_derived.learning_power:.4f} acc/s")
            winner = "TEP" if tep_derived.learning_power > bp_derived.learning_power else "BP"
            print(f"        Winner: {winner}")
            
            print(f"      Parameter Efficiency (acc/log10(params)):")
            print(f"        TEP: {tep_derived.param_efficiency:.4f}")
            print(f"        BP:  {bp_derived.param_efficiency:.4f}")
            winner = "TEP" if tep_derived.param_efficiency > bp_derived.param_efficiency else "BP"
            print(f"        Winner: {winner}")
            
            print(f"      Overall Efficiency Score:")
            print(f"        TEP: {tep_derived.efficiency_score:.4f}")
            print(f"        BP:  {bp_derived.efficiency_score:.4f}")
            winner = "TEP" if tep_derived.efficiency_score > bp_derived.efficiency_score else "BP"
            print(f"        Winner: {winner}")
            
            # Print hyperparameters
            print(f"\n   ⚙️  Best Hyperparameters:")
            print(f"      TEP: {summarize_config(tep_best.config, 'tep')}")
            print(f"      BP:  {summarize_config(bp_best.config, 'bp')}")
        
        # Print top 3 configs from each
        print(f"\n   🏆 Top 3 TEP Configs (by accuracy):")
        tep_sorted = sorted(
            [t for t in tep_trials if t.status == "complete" and t.accuracy > 0],
            key=lambda x: x.accuracy,
            reverse=True
        )[:3]
        for i, trial in enumerate(tep_sorted, 1):
            derived = compute_derived_metrics(
                trial.accuracy, trial.wall_time_seconds,
                trial.param_count, trial.convergence_steps
            )
            print(f"      #{i}: acc={trial.accuracy:.4f}, power={derived.learning_power:.4f}, eff={derived.efficiency_score:.4f}")
            print(f"          {summarize_config(trial.config, 'tep')}")
        
        print(f"\n   🏆 Top 3 BP Configs (by accuracy):")
        bp_sorted = sorted(
            [t for t in bp_trials if t.status == "complete" and t.accuracy > 0],
            key=lambda x: x.accuracy,
            reverse=True
        )[:3]
        for i, trial in enumerate(bp_sorted, 1):
            derived = compute_derived_metrics(
                trial.accuracy, trial.wall_time_seconds,
                trial.param_count, trial.convergence_steps
            )
            print(f"      #{i}: acc={trial.accuracy:.4f}, power={derived.learning_power:.4f}, eff={derived.efficiency_score:.4f}")
            print(f"          {summarize_config(trial.config, 'bp')}")
        
        # Print winner
        print(f"\n   🎖️  Winner: {comparison['winner']}")
        if "tie_broken_by" in comparison:
            print(f"       (tie broken by {comparison['tie_broken_by']})")
