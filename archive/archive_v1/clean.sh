#!/bin/bash
# Clean the research output database

OUTPUT_DIR="research_output"
DB_FILE="$OUTPUT_DIR/results.db"
JSON_FILE="$OUTPUT_DIR/results.json"

if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  This will delete all data in $OUTPUT_DIR"
    read -p "Are you sure? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
        echo "✅ Database wiped. Ready for fresh campaign."
    else
        echo "Cancelled."
    fi
else
    mkdir -p "$OUTPUT_DIR"
    echo "✨ Created new output directory."
fi
