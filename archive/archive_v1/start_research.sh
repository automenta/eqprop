#!/bin/bash
#
# TorEqProp Autonomous Research Launcher
#
# Turn-key automation: INPUT = Time + Electricity -> OUTPUT = Beneficial Research
#
# Usage:
#   ./start_research.sh              # Run indefinitely
#   ./start_research.sh 8            # Run for 8 hours
#   ./start_research.sh resume       # Resume from checkpoint
#   ./start_research.sh status       # Show status
#

set -e

cd "$(dirname "$0")"

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║     🔬 TorEqProp AUTONOMOUS RESEARCH SYSTEM 🔬                    ║"
echo "║                                                                   ║"
echo "║     INPUT:  Time + Electricity                                    ║"
echo "║     OUTPUT: Beneficial Research                                   ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo

case "${1:-run}" in
    status)
        python autonomous_researcher.py --status
        ;;
    resume)
        echo "📂 Resuming from checkpoint..."
        python autonomous_researcher.py --resume
        ;;
    [0-9]*)
        echo "⏰ Running for $1 hours..."
        python autonomous_researcher.py --hours "$1"
        ;;
    run|*)
        echo "🔄 Running indefinitely (Ctrl+C to stop safely)..."
        python autonomous_researcher.py
        ;;
esac
