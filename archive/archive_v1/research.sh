#!/bin/bash
#
# Scientific Research Engine Launcher
#
# Usage:
#   ./research.sh              # Quick 5-minute validation
#   ./research.sh turbo        # Quick 3-minute turbo mode  
#   ./research.sh campaign     # Full progressive campaign
#   ./research.sh analyze      # Analyze hyperparameter importance
#

cd "$(dirname "$0")"

echo "🔬 TorEqProp Scientific Research Engine"
echo

case "${1:-quick}" in
    turbo|fast)
        echo "⚡ TURBO MODE - 3 minute quick validation"
        python research_engine.py --quick --minutes 3
        ;;
    campaign|full)
        echo "📊 FULL CAMPAIGN - Progressive validation"
        python research_engine.py --campaign
        ;;
    analyze)
        echo "📈 ANALYZING hyperparameter importance"
        python research_engine.py --analyze
        ;;
    [0-9]*)
        echo "⏱️ Running for $1 minutes"
        python research_engine.py --quick --minutes "$1"
        ;;
    quick|*)
        echo "🚀 QUICK MODE - 5 minute validation"
        python research_engine.py --quick --minutes 5
        ;;
esac
