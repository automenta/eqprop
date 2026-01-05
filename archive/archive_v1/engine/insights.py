"""Insight Engine - Real-time analysis and hypothesis validation."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import numpy as np
from scipy import stats

from .database import Trial, Insight, ExperimentDB


@dataclass
class ClaimValidation:
    """Result of validating a research claim."""
    claim: str
    is_supported: bool
    confidence: float  # 0-1
    p_value: Optional[float]
    effect_size: Optional[float]
    evidence_summary: str
    skeptic_concerns: List[str]
    suggestions: List[str]


class InsightEngine:
    """Generates insights from experimental results.
    
    Continuously analyzes completed trials to:
    - Identify patterns and trends
    - Test research hypotheses
    - Generate publication-ready findings
    - Anticipate academic skepticism
    """
    
    def __init__(self, db: ExperimentDB):
        self.db = db
    
    def analyze_trial(self, trial: Trial) -> List[Insight]:
        """Generate insights from a newly completed trial."""
        insights = []
        
        # Check for notable performance
        if trial.performance > 0.95:
            insights.append(Insight(
                insight_id=str(uuid.uuid4())[:8],
                insight_type="finding",
                title=f"High performance on {trial.task}",
                description=f"{trial.algorithm} achieved {trial.performance:.2%} on {trial.task}",
                evidence=[trial.trial_id],
                confidence=0.8
            ))
        
        # Check for speed anomalies
        similar_trials = self.db.get_trials(task=trial.task, status="complete")
        if similar_trials:
            avg_time = np.mean([t.wall_time_seconds for t in similar_trials])
            if trial.wall_time_seconds < avg_time * 0.5:
                insights.append(Insight(
                    insight_id=str(uuid.uuid4())[:8],
                    insight_type="finding",
                    title=f"Fast trial on {trial.task}",
                    description=f"{trial.algorithm} ran {avg_time/trial.wall_time_seconds:.1f}x faster than average",
                    evidence=[trial.trial_id],
                    confidence=0.7
                ))
        
        return insights
    
    def compare_algorithms(self, task: str) -> Optional[Insight]:
        """Compare EqProp vs BP on a specific task."""
        eqprop_trials = self.db.get_trials(algorithm="eqprop", task=task, status="complete")
        bp_trials = self.db.get_trials(algorithm="bp", task=task, status="complete")
        
        if len(eqprop_trials) < 2 or len(bp_trials) < 2:
            return None
        
        eq_perfs = [t.performance for t in eqprop_trials]
        bp_perfs = [t.performance for t in bp_trials]
        
        eq_times = [t.wall_time_seconds for t in eqprop_trials]
        bp_times = [t.wall_time_seconds for t in bp_trials]
        
        # Statistical comparison
        t_stat, p_value = stats.ttest_ind(eq_perfs, bp_perfs)
        eq_mean, bp_mean = np.mean(eq_perfs), np.mean(bp_perfs)
        eq_time_mean, bp_time_mean = np.mean(eq_times), np.mean(bp_times)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(eq_perfs)**2 + np.std(bp_perfs)**2) / 2)
        cohens_d = (eq_mean - bp_mean) / pooled_std if pooled_std > 0 else 0
        
        # Generate insight
        if eq_mean > bp_mean and p_value < 0.05:
            winner = "EqProp"
            diff = eq_mean - bp_mean
        elif bp_mean > eq_mean and p_value < 0.05:
            winner = "BP"
            diff = bp_mean - eq_mean
        else:
            winner = "Neither (not significant)"
            diff = abs(eq_mean - bp_mean)
        
        time_ratio = bp_time_mean / eq_time_mean if eq_time_mean > 0 else 1
        
        return Insight(
            insight_id=str(uuid.uuid4())[:8],
            insight_type="comparison",
            title=f"{task}: {winner} wins",
            description=(
                f"EqProp: {eq_mean:.2%} ± {np.std(eq_perfs):.2%} ({eq_time_mean:.1f}s)\n"
                f"BP: {bp_mean:.2%} ± {np.std(bp_perfs):.2%} ({bp_time_mean:.1f}s)\n"
                f"p-value: {p_value:.4f}, Cohen's d: {cohens_d:.2f}\n"
                f"Speed ratio: EqProp is {time_ratio:.1f}x {'faster' if time_ratio > 1 else 'slower'}"
            ),
            evidence=[t.trial_id for t in eqprop_trials + bp_trials],
            confidence=1 - p_value
        )
    
    def validate_hypothesis(self, hypothesis_name: str) -> ClaimValidation:
        """Rigorously test a research hypothesis.
        
        Applies academic skepticism to our own claims.
        """
        if hypothesis_name == "speed_advantage":
            return self._validate_speed_advantage()
        elif hypothesis_name == "rl_superiority":
            return self._validate_rl_superiority()
        elif hypothesis_name == "optimal_beta":
            return self._validate_optimal_beta()
        else:
            return ClaimValidation(
                claim=hypothesis_name,
                is_supported=False,
                confidence=0.0,
                p_value=None,
                effect_size=None,
                evidence_summary="Unknown hypothesis",
                skeptic_concerns=["Hypothesis not defined"],
                suggestions=["Define hypothesis test"]
            )
    
    def _validate_speed_advantage(self) -> ClaimValidation:
        """Test: EqProp achieves 90% of BP accuracy at 2x+ speed."""
        all_trials = self.db.get_trials(status="complete")
        
        tasks_with_both = set()
        for trial in all_trials:
            if trial.algorithm == "eqprop":
                bp_exists = any(t.task == trial.task and t.algorithm == "bp" for t in all_trials)
                if bp_exists:
                    tasks_with_both.add(trial.task)
        
        if not tasks_with_both:
            return ClaimValidation(
                claim="speed_advantage",
                is_supported=False,
                confidence=0.0,
                p_value=None,
                effect_size=None,
                evidence_summary="Insufficient data: no tasks with both EqProp and BP trials",
                skeptic_concerns=["Need matched comparisons"],
                suggestions=["Run both algorithms on same tasks"]
            )
        
        speed_ratios = []
        accuracy_ratios = []
        
        for task in tasks_with_both:
            eq_trials = [t for t in all_trials if t.task == task and t.algorithm == "eqprop"]
            bp_trials = [t for t in all_trials if t.task == task and t.algorithm == "bp"]
            
            eq_best = max(eq_trials, key=lambda t: t.performance)
            bp_best = max(bp_trials, key=lambda t: t.performance)
            
            if bp_best.wall_time_seconds > 0 and bp_best.performance > 0:
                speed_ratios.append(bp_best.wall_time_seconds / eq_best.wall_time_seconds)
                accuracy_ratios.append(eq_best.performance / bp_best.performance)
        
        avg_speed = np.mean(speed_ratios) if speed_ratios else 0
        avg_acc_ratio = np.mean(accuracy_ratios) if accuracy_ratios else 0
        
        is_supported = avg_speed >= 2.0 and avg_acc_ratio >= 0.9
        
        return ClaimValidation(
            claim="EqProp achieves 90% of BP accuracy at 2x+ speed",
            is_supported=is_supported,
            confidence=min(avg_acc_ratio, 1.0) if is_supported else avg_acc_ratio,
            p_value=None,  # Would need proper statistical test
            effect_size=avg_speed,
            evidence_summary=(
                f"Across {len(tasks_with_both)} tasks:\n"
                f"- Average speed ratio: {avg_speed:.1f}x\n"
                f"- Average accuracy ratio: {avg_acc_ratio:.2%}"
            ),
            skeptic_concerns=[
                "Are model sizes matched?",
                "Is this due to fewer epochs, not algorithmic difference?",
                "What about variance across seeds?",
            ] if not is_supported else [],
            suggestions=[
                "Run with matched d_model for both algorithms",
                "Report confidence intervals",
                "Test on more tasks",
            ]
        )
    
    def _validate_rl_superiority(self) -> ClaimValidation:
        """Test: EqProp outperforms BP on RL tasks."""
        rl_tasks = ["cartpole", "acrobot", "lunarlander", "mountaincar"]
        
        eqprop_wins = 0
        tasks_tested = 0
        
        for task in rl_tasks:
            eq_trials = self.db.get_trials(algorithm="eqprop", task=task, status="complete")
            bp_trials = self.db.get_trials(algorithm="bp", task=task, status="complete")
            
            if eq_trials and bp_trials:
                tasks_tested += 1
                eq_best = max(t.performance for t in eq_trials)
                bp_best = max(t.performance for t in bp_trials)
                
                if eq_best > bp_best:
                    eqprop_wins += 1
        
        is_supported = eqprop_wins >= 2 and tasks_tested >= 2
        
        return ClaimValidation(
            claim="EqProp outperforms BP on RL tasks",
            is_supported=is_supported,
            confidence=eqprop_wins / tasks_tested if tasks_tested > 0 else 0,
            p_value=None,
            effect_size=eqprop_wins,
            evidence_summary=f"EqProp wins on {eqprop_wins}/{tasks_tested} RL tasks",
            skeptic_concerns=[
                "RL results have high variance",
                "Need multiple seeds per environment",
                "Hyperparameters may not be tuned for BP",
            ],
            suggestions=[
                "Run 5+ seeds per environment",
                "Report confidence intervals",
                "Try more RL environments",
            ]
        )
    
    def _validate_optimal_beta(self) -> ClaimValidation:
        """Test: β=0.22 is optimal for transformer training."""
        eq_trials = self.db.get_trials(algorithm="eqprop", status="complete")
        
        beta_results: Dict[float, List[float]] = {}
        for trial in eq_trials:
            beta = trial.config.get("beta", 0.1)
            if beta not in beta_results:
                beta_results[beta] = []
            beta_results[beta].append(trial.performance)
        
        if len(beta_results) < 3:
            return ClaimValidation(
                claim="β=0.22 is optimal",
                is_supported=False,
                confidence=0.0,
                p_value=None,
                effect_size=None,
                evidence_summary=f"Only {len(beta_results)} β values tested",
                skeptic_concerns=["Need more β values to establish optimum"],
                suggestions=["Run β sweep: 0.18, 0.20, 0.22, 0.24, 0.26"]
            )
        
        # Find best β
        avg_by_beta = {b: np.mean(perfs) for b, perfs in beta_results.items()}
        best_beta = max(avg_by_beta, key=avg_by_beta.get)
        
        is_022_optimal = abs(best_beta - 0.22) < 0.02
        
        return ClaimValidation(
            claim="β=0.22 is optimal",
            is_supported=is_022_optimal,
            confidence=0.8 if is_022_optimal else 0.3,
            p_value=None,
            effect_size=avg_by_beta.get(0.22, 0),
            evidence_summary=f"Best β: {best_beta} (avg perf: {avg_by_beta[best_beta]:.2%})",
            skeptic_concerns=[
                "Is 0.22 significantly better than neighbors?",
                "Does optimal β vary by task?",
                "What about β × d_model interaction?",
            ],
            suggestions=[
                "Run fine-grained β sweep",
                "Test multiple tasks",
                "Report statistical significance of peak",
            ]
        )
    
    def generate_skeptic_response(self, claim: str) -> str:
        """Generate likely academic criticism of a claim.
        
        Helps prepare for peer review.
        """
        skeptic_responses = {
            "speed_advantage": [
                "Speed comparisons are meaningless without matched compute budgets.",
                "Different model sizes confound the comparison.",
                "What happens with optimal hyperparameters for both?",
            ],
            "rl_superiority": [
                "RL results are notoriously noisy - need many seeds.",
                "Policy gradient methods work fine with standard autodiff.",
                "What about off-policy methods like SAC?",
            ],
            "optimal_beta": [
                "A single optimal β seems unlikely - it should depend on architecture.",
                "How sensitive are results to this parameter?",
                "What's the theoretical justification for β > 0?",
            ],
        }
        
        return "\n".join(skeptic_responses.get(claim, ["Claim not addressed"]))
    
    def generate_discoveries_markdown(self) -> str:
        """Generate a markdown summary of all discoveries."""
        insights = self.db.get_insights()
        summary = self.db.get_summary()
        
        md = "# TorEqProp Discoveries\n\n"
        md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
        
        md += "## Summary\n\n"
        md += f"- Total trials: {summary['total_trials']}\n"
        md += f"- Completed: {summary['status_counts'].get('complete', 0)}\n"
        md += f"- Insights generated: {summary['insights_count']}\n\n"
        
        if summary.get('best_results'):
            md += "## Best Results by Task\n\n"
            md += "| Task | Algorithm | Performance |\n"
            md += "|------|-----------|-------------|\n"
            for result in summary['best_results']:
                md += f"| {result['task']} | {result['algorithm']} | {result['performance']:.4f} |\n"
            md += "\n"
        
        if insights:
            md += "## Key Insights\n\n"
            for insight in sorted(insights, key=lambda i: i.confidence, reverse=True)[:10]:
                md += f"### {insight.title}\n\n"
                md +=  f"{insight.description}\n\n"
                md += f"*Confidence: {insight.confidence:.0%}*\n\n"
        
        return md
