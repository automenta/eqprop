#!/usr/bin/env python3
"""
README Auto-Updater - Keeps README.md current with validated results.

Updates specific sections marked with HTML comments:
<!-- VALIDATION_CLAIMS_START --> ... <!-- VALIDATION_CLAIMS_END -->
<!-- VALIDATION_RESULTS_START --> ... <!-- VALIDATION_RESULTS_END -->
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from statistics import ComparisonResult


class ReadmeUpdater:
    """Auto-update README.md with validation results."""
    
    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = Path(readme_path)
    
    def update(self, 
               results: Dict[str, ComparisonResult],
               summary: Dict[str, any]) -> bool:
        """Update README with latest validated results."""
        if not self.readme_path.exists():
            print(f"Warning: README not found at {self.readme_path}")
            return False
        
        content = self.readme_path.read_text()
        
        # Update claims table
        claims_section = self._generate_claims_section(results)
        content = self._replace_section(content, "VALIDATION_CLAIMS", claims_section)
        
        # Update detailed results
        results_section = self._generate_results_section(results, summary)
        content = self._replace_section(content, "VALIDATION_RESULTS", results_section)
        
        self.readme_path.write_text(content)
        return True
    
    def _replace_section(self, content: str, section_name: str, new_content: str) -> str:
        """Replace content between markers."""
        start_marker = f"<!-- {section_name}_START -->"
        end_marker = f"<!-- {section_name}_END -->"
        
        pattern = re.compile(
            f"({re.escape(start_marker)}).*?({re.escape(end_marker)})",
            re.DOTALL
        )
        
        if pattern.search(content):
            replacement = f"{start_marker}\n{new_content}\n{end_marker}"
            return pattern.sub(replacement, content)
        
        # Markers don't exist, add to end
        return content
    
    def _generate_claims_section(self, results: Dict[str, ComparisonResult]) -> str:
        """Generate claims table markdown."""
        lines = [
            "",
            "| Environment | EqProp | BP | Improvement | Status |",
            "|-------------|--------|-----|-------------|--------|"
        ]
        
        for env, result in sorted(results.items()):
            status = "✅ **VALIDATED**" if result.is_breakthrough else \
                     "✅ Significant" if result.is_significant else "⚠️ Pending"
            
            improvement = f"+{result.improvement_pct:.0f}%{result.significance_stars}"
            
            lines.append(
                f"| {env} | {result.algo1_mean:.0f}±{result.algo1_std:.0f} | "
                f"{result.algo2_mean:.0f}±{result.algo2_std:.0f} | "
                f"{improvement} | {status} |"
            )
        
        lines.append("")
        lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
        lines.append("")
        
        return "\n".join(lines)
    
    def _generate_results_section(self, 
                                   results: Dict[str, ComparisonResult],
                                   summary: Dict) -> str:
        """Generate detailed results markdown."""
        lines = [
            "",
            "### Statistical Validation Details",
            ""
        ]
        
        for env, result in sorted(results.items()):
            lines.extend([
                f"#### {env}",
                "",
                f"- **EqProp**: {result.algo1_mean:.1f} ± {result.algo1_std:.1f} (n={result.algo1_n})",
                f"- **BP**: {result.algo2_mean:.1f} ± {result.algo2_std:.1f} (n={result.algo2_n})",
                f"- **Improvement**: {result.improvement_pct:+.1f}%",
                f"- **p-value**: {result.p_value:.4f}",
                f"- **Cohen's d**: {result.cohens_d:.2f} ({'large' if result.is_large_effect else 'medium' if abs(result.cohens_d) > 0.5 else 'small'} effect)",
                f"- **95% CI**: [{result.ci_lower:.1f}, {result.ci_upper:.1f}]",
                ""
            ])
        
        # Add summary
        total = summary.get("total_experiments", 0)
        completed = summary.get("completed", 0)
        breakthroughs = sum(1 for r in results.values() if r.is_breakthrough)
        
        lines.extend([
            "### Summary",
            "",
            f"- Total experiments: {completed}/{total}",
            f"- Breakthroughs validated: {breakthroughs}/{len(results)}",
            ""
        ])
        
        return "\n".join(lines)
    
    def add_markers_if_missing(self):
        """Add section markers to README if they don't exist."""
        if not self.readme_path.exists():
            return
        
        content = self.readme_path.read_text()
        
        # Check for claims section
        if "<!-- VALIDATION_CLAIMS_START -->" not in content:
            # Find a good place to insert (after Key Discoveries or at end)
            insert_point = content.find("## Key Discoveries")
            if insert_point == -1:
                insert_point = len(content)
            else:
                # Find end of Key Discoveries section
                next_section = content.find("\n## ", insert_point + 1)
                if next_section != -1:
                    insert_point = next_section
            
            claims_block = """
---

## Validated Results

<!-- VALIDATION_CLAIMS_START -->
*Validation in progress...*
<!-- VALIDATION_CLAIMS_END -->

<!-- VALIDATION_RESULTS_START -->
<!-- VALIDATION_RESULTS_END -->

"""
            content = content[:insert_point] + claims_block + content[insert_point:]
            self.readme_path.write_text(content)
            print("Added validation markers to README.md")


# Self-test
if __name__ == "__main__":
    print("Testing ReadmeUpdater...")
    
    # Create mock comparison result
    from statistics import ComparisonResult
    
    mock_result = ComparisonResult(
        algo1_name="EqProp",
        algo2_name="BP",
        algo1_mean=341.2,
        algo1_std=23.4,
        algo1_n=10,
        algo2_mean=187.3,
        algo2_std=31.2,
        algo2_n=10,
        difference=153.9,
        improvement_pct=82.1,
        t_statistic=12.5,
        p_value=0.00001,
        ci_lower=130.0,
        ci_upper=178.0,
        cohens_d=2.34,
        is_significant=True,
        is_large_effect=True,
        is_breakthrough=True,
        algo1_wins=True
    )
    
    updater = ReadmeUpdater()
    claims = updater._generate_claims_section({"CartPole-v1": mock_result})
    print(claims)
    
    print("\n✅ ReadmeUpdater tests passed!")
