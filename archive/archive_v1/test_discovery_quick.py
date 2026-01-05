#!/usr/bin/env python
"""Quick test of discovery engine."""

from engine.orchestrator import DiscoveryOrchestrator
from engine.config import RAPID_CONFIG

# Configure for quick test
RAPID_CONFIG.max_total_time_hours = 0.02  # ~1 minute
RAPID_CONFIG.max_trial_time_seconds = 30

# Run test campaign
orch = DiscoveryOrchestrator(RAPID_CONFIG, db_path='results/test_campaign.db')
result = orch.run_campaign(patience_hours=0.02)

print("\nTest Result:", result)
