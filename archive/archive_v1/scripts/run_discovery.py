#!/usr/bin/env python
"""TorEqProp Discovery Engine - Main entry point.

Run discovery campaign:
    python run_discovery.py --hours 1

Check status:
    python run_discovery.py --status
    
Validate hypotheses:
    python run_discovery.py --validate
"""

from engine.orchestrator import main

if __name__ == "__main__":
    main()
