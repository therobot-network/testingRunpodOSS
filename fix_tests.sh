#!/bin/bash

# Quick fix for the test script filename issue
# Replace the problematic line in run_basic_tests.sh

echo "🔧 Fixing test script filename issue..."

# Fix the log file naming issue
sed -i 's/local log_file="$LOG_DIR\/${test_name}_${model\/\/:\//}_$TIMESTAMP.log"/local model_safe=${model\/\/:/}; local log_file="$LOG_DIR\/${test_name}_${model_safe}_$TIMESTAMP.log"/' run_basic_tests.sh

echo "✅ Fixed! You can now run ./run_basic_tests.sh"
