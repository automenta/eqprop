#!/bin/bash
#
# Quick Rapid Research Launcher
#
# Gets tangible results in MINUTES with a beautiful TUI dashboard!
#

cd "$(dirname "$0")"

echo "🚀 Launching Rapid Research Dashboard..."
echo

case "${1:-normal}" in
    turbo|fast)
        echo "⚡ TURBO MODE - Ultra-fast with tiny models"
        python rapid_dashboard.py --turbo --minutes "${2:-3}"
        ;;
    [0-9]*)
        echo "⏱️ Running for $1 minutes"
        python rapid_dashboard.py --minutes "$1"
        ;;
    *)
        echo "📊 Standard mode - 5 minutes"
        python rapid_dashboard.py --minutes 5
        ;;
esac
