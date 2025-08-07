#!/bin/bash

# Basic CLI tests for GPT-OSS models
# Run this script to execute a series of tests on the GPT-OSS models

set -e

MODEL_20B="gpt-oss:20b"
MODEL_120B="gpt-oss:120b"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

echo "🧪 Starting GPT-OSS CLI Tests - $TIMESTAMP"

# Check if models are available
echo "🔍 Checking available models..."
ollama list

# Function to run a test with timing and logging
run_test() {
    local model=$1
    local prompt=$2
    local test_name=$3
    local log_file="$LOG_DIR/${test_name}_${model//://}_$TIMESTAMP.log"
    
    echo "▶️  Running test: $test_name with $model"
    echo "📝 Logging to: $log_file"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the test
    echo "=== Test: $test_name ===" > "$log_file"
    echo "Model: $model" >> "$log_file"
    echo "Timestamp: $(date)" >> "$log_file"
    echo "Prompt: $prompt" >> "$log_file"
    echo "=== Response ===" >> "$log_file"
    
    ollama run "$model" "$prompt" >> "$log_file" 2>&1
    
    # Record end time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Duration: ${duration}s" >> "$log_file"
    echo "✅ Completed in ${duration}s"
    echo ""
}

# Test 1: Basic reasoning
echo "🧠 Test 1: Basic Reasoning"
run_test "$MODEL_20B" "Explain the concept of recursion in programming with a simple example." "basic_reasoning"

# Test 2: Code generation
echo "💻 Test 2: Code Generation"
run_test "$MODEL_20B" "Write a Python function that finds the longest common subsequence between two strings. Include comments and example usage." "code_generation"

# Test 3: Problem solving
echo "🔧 Test 3: Problem Solving"
run_test "$MODEL_20B" "I have a list of integers [3, 1, 4, 1, 5, 9, 2, 6]. I need to find all pairs that sum to 10. Show me step by step how to solve this efficiently." "problem_solving"

# Test 4: Chain of thought reasoning
echo "🔗 Test 4: Chain of Thought"
run_test "$MODEL_20B" "A farmer has chickens and rabbits. In total, there are 35 heads and 94 legs. How many chickens and how many rabbits are there? Show your reasoning step by step." "chain_of_thought"

# Test 5: Structured output
echo "📋 Test 5: Structured Output"
run_test "$MODEL_20B" "Create a JSON object describing a fictional book with the following fields: title, author, genre, publication_year, summary, and rating. Make it realistic and interesting." "structured_output"

# GPU Memory check after tests
echo "🎯 GPU Memory Status After Tests:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# Summary
echo "📊 Test Summary:"
echo "   - All tests completed and logged to $LOG_DIR/"
echo "   - Use 'ls -la $LOG_DIR/' to see all log files"
echo "   - Run 'python3 scripts/analyze_results.py' to analyze performance"

echo "🎉 Basic tests completed successfully!"
