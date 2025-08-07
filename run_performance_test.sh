#!/bin/bash

# GPT-OSS Performance Benchmark Script
# Tests TTFT, tokens/second, and GPU performance using real prompts

set -e

MODEL="gpt-oss:20b"
DATA_FILE="data/101-200.csv"
NUM_TESTS=5
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 GPT-OSS Performance Benchmark${NC}"
echo -e "${CYAN}Model: $MODEL${NC}"
echo -e "${CYAN}Data: $DATA_FILE${NC}"
echo -e "${CYAN}Tests: $NUM_TESTS${NC}"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}❌ Error: Data file $DATA_FILE not found${NC}"
    exit 1
fi

# Check if model is available
echo -e "${YELLOW}🔍 Checking if model is available...${NC}"
if ! ollama list | grep -q "$MODEL"; then
    echo -e "${RED}❌ Model $MODEL not found. Available models:${NC}"
    ollama list
    exit 1
fi

# Create logs directory
mkdir -p $LOG_DIR

# Check GPU status
echo -e "${YELLOW}🎯 GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi
echo ""

# Function to run a single performance test
run_performance_test() {
    local prompt="$1"
    local test_num="$2"
    local question="$3"
    
    echo -e "${CYAN}📊 Test $test_num: ${question:0:60}...${NC}"
    
    # Create log file
    local log_file="$LOG_DIR/perf_test_${test_num}_$TIMESTAMP.log"
    
    # Record start time (nanoseconds for precision)
    local start_time=$(date +%s.%N)
    
    # Run inference and capture output
    echo "=== Performance Test $test_num ===" > "$log_file"
    echo "Question: $question" >> "$log_file"
    echo "Model: $MODEL" >> "$log_file"
    echo "Start Time: $(date)" >> "$log_file"
    echo "=== Response ===" >> "$log_file"
    
    # Run the actual test with timing
    if timeout 300 ollama run "$MODEL" "$prompt" >> "$log_file" 2>&1; then
        local end_time=$(date +%s.%N)
        local total_time=$(echo "$end_time - $start_time" | bc)
        
        # Estimate tokens (rough approximation)
        local response_chars=$(wc -c < "$log_file")
        local estimated_tokens=$(echo "scale=0; $response_chars / 4" | bc)
        local tokens_per_second=$(echo "scale=2; $estimated_tokens / $total_time" | bc)
        
        echo "=== Performance Metrics ===" >> "$log_file"
        echo "Total Time: ${total_time}s" >> "$log_file"
        echo "Estimated Tokens: $estimated_tokens" >> "$log_file"
        echo "Tokens/Second: $tokens_per_second" >> "$log_file"
        
        echo -e "${GREEN}✅ Completed in ${total_time}s | ~$tokens_per_second tokens/sec${NC}"
        
        # Record GPU status after test
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status After:" >> "$log_file"
            nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits >> "$log_file"
        fi
        
        return 0
    else
        echo -e "${RED}❌ Test failed or timed out${NC}"
        echo "Status: FAILED" >> "$log_file"
        return 1
    fi
}

# Read prompts from CSV and run tests
echo -e "${YELLOW}📝 Loading prompts from $DATA_FILE...${NC}"

# Skip header and get random lines
test_count=0
successful_tests=0
total_time=0

# Use Python to extract random prompts from CSV
python3 -c "
import csv
import random
import sys

try:
    with open('$DATA_FILE', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Select random prompts
    selected = random.sample(rows, min($NUM_TESTS, len(rows)))
    
    for i, row in enumerate(selected, 1):
        question = row.get('User Search Question', 'No question')
        prompt = row.get('full_prompt', '')
        
        if prompt:
            # Write to temp files for bash to read
            with open(f'/tmp/prompt_{i}.txt', 'w') as pf:
                pf.write(prompt)
            with open(f'/tmp/question_{i}.txt', 'w') as qf:
                qf.write(question)
            print(f'{i}')
        
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" > /tmp/test_numbers.txt

if [ ! -s /tmp/test_numbers.txt ]; then
    echo -e "${RED}❌ Failed to load prompts${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prompts loaded successfully${NC}"
echo ""

# Run the tests
while read test_num; do
    if [ -f "/tmp/prompt_${test_num}.txt" ] && [ -f "/tmp/question_${test_num}.txt" ]; then
        prompt=$(cat "/tmp/prompt_${test_num}.txt")
        question=$(cat "/tmp/question_${test_num}.txt")
        
        test_count=$((test_count + 1))
        
        if run_performance_test "$prompt" "$test_num" "$question"; then
            successful_tests=$((successful_tests + 1))
        fi
        
        echo ""
        
        # Small delay between tests
        sleep 2
    fi
done < /tmp/test_numbers.txt

# Clean up temp files
rm -f /tmp/prompt_*.txt /tmp/question_*.txt /tmp/test_numbers.txt

# Summary
echo -e "${BLUE}📊 Performance Test Summary${NC}"
echo -e "${GREEN}✅ Successful tests: $successful_tests/$test_count${NC}"
echo -e "${CYAN}📁 Logs saved to: $LOG_DIR/perf_test_*_$TIMESTAMP.log${NC}"

# Final GPU status
echo -e "${YELLOW}🎯 Final GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
fi

echo ""
echo -e "${GREEN}🎉 Performance benchmark completed!${NC}"
echo -e "${CYAN}💡 Run 'python3 scripts/performance_benchmark.py' for detailed analysis${NC}"
