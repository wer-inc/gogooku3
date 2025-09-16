#!/bin/bash
# Output directories cleanup and organization script

echo "=== Output Directories Organization Plan ==="
echo

# 1. Create organized structure
echo "1. Creating organized directory structure..."
mkdir -p output_organized/{datasets,experiments,tests,benchmarks,archives}

# 2. Move main datasets
echo "2. Moving main ML datasets..."
if [ -d "output" ]; then
    # Keep only latest and important datasets
    find output -name "ml_dataset_*.parquet" -size +100M | while read file; do
        basename=$(basename "$file")
        echo "  - Moving $basename to datasets/"
        # cp "$file" output_organized/datasets/
    done
fi

# 3. Archive old experiments
echo "3. Archiving old test outputs..."
for dir in output_smoke output_offline_full output_offline_full2 test_output_full test_output_base; do
    if [ -d "$dir" ]; then
        echo "  - Archiving $dir"
        # tar -czf "output_organized/archives/${dir}_$(date +%Y%m%d).tar.gz" "$dir"
        # rm -rf "$dir"
    fi
done

# 4. Organize experiment outputs
echo "4. Organizing experiment outputs..."
if [ -d "outputs/2025-09-15" ]; then
    echo "  - Moving 2025-09-15 experiments"
    # mv outputs/2025-09-15 output_organized/experiments/
fi

# 5. Move benchmark data
echo "5. Moving benchmark outputs..."
if [ -d "benchmark_output" ]; then
    echo "  - Moving benchmark_output"
    # mv benchmark_output output_organized/benchmarks/
fi

# 6. Clean up symlinks
echo "6. Cleaning up broken symlinks..."
find output -type l ! -exec test -e {} \; -print 2>/dev/null | while read link; do
    echo "  - Removing broken symlink: $link"
    # rm "$link"
done

echo
echo "=== Summary ==="
echo "Organized structure:"
echo "  output_organized/"
echo "    ├── datasets/      # Main ML datasets"
echo "    ├── experiments/   # Experiment results"
echo "    ├── tests/         # Test outputs"
echo "    ├── benchmarks/    # Benchmark results"
echo "    └── archives/      # Archived old outputs"
echo
echo "To execute cleanup, uncomment the actual commands in this script."
