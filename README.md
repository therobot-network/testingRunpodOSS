# GPT-OSS Testing Suite for RunPod

A comprehensive testing framework for OpenAI's GPT-OSS models running on Ollama, specifically optimized for RunPod RTX 4090 GPU instances.

## 🚀 Quick Start

### 1. Run the Setup Script
```bash
chmod +x setup.sh
sudo ./setup.sh
```

This will:
- Install Ollama and required dependencies
- Pull the GPT-OSS 20B model (recommended for RTX 4090)
- Install Python dependencies
- Configure the environment

### 2. Verify Installation
```bash
# Check Ollama is running
ollama list

# Check GPU status
nvidia-smi

# Test basic functionality
ollama run gpt-oss:20b "Hello, how are you?"
```

### 3. Run Your First Tests
```bash
# Quick basic tests
./run_basic_tests.sh

# Interactive testing
python3 scripts/interactive_test.py

# Monitor GPU during tests
python3 scripts/gpu_monitor.py --log-file logs/gpu_usage.log
```

## 📁 Project Structure

```
testingRunpod/
├── setup.sh                    # Automated setup script
├── run_basic_tests.sh          # Basic CLI test suite
├── requirements.txt            # Python dependencies
├── config/
│   ├── test_scenarios.json     # Test scenarios and configurations
│   └── ollama_config.json      # Ollama and GPU settings
├── scripts/
│   ├── interactive_test.py     # Interactive testing interface
│   ├── analyze_results.py      # Results analysis and visualization
│   └── gpu_monitor.py          # Real-time GPU monitoring
└── logs/                       # Test results and logs (created automatically)
```

## 🧪 Testing Options

### Basic CLI Tests
```bash
./run_basic_tests.sh
```
Runs a series of predefined tests covering:
- Basic reasoning
- Code generation  
- Problem solving
- Chain of thought reasoning
- Structured output

### Interactive Testing
```bash
python3 scripts/interactive_test.py
```
Features:
- Model selection (20B/120B)
- Custom prompts
- Configurable reasoning effort (low/medium/high)
- Built-in benchmark suite
- Test history tracking

### Advanced Options
```bash
# Run with specific model
python3 scripts/interactive_test.py --model gpt-oss:20b

# Auto-run benchmark suite
python3 scripts/interactive_test.py --auto-benchmark

# Analyze previous test results
python3 scripts/analyze_results.py --plots --export
```

## 📊 Monitoring & Analysis

### GPU Monitoring
```bash
# Real-time monitoring
python3 scripts/gpu_monitor.py

# Log metrics for analysis
python3 scripts/gpu_monitor.py --log-file logs/gpu_metrics.log --duration 300
```

### Results Analysis
```bash
# Generate summary and visualizations
python3 scripts/analyze_results.py --plots --export

# Export data to CSV
python3 scripts/analyze_results.py --export
```

## ⚙️ Configuration

### Model Recommendations for RTX 4090 (24GB VRAM)

**GPT-OSS 20B (Recommended)**
- Memory usage: ~14GB
- Perfect fit for RTX 4090
- Fast inference times
- Good for most testing scenarios

**GPT-OSS 120B (Advanced)**  
- Memory usage: ~65GB
- Requires CPU offloading on RTX 4090
- Slower but higher quality responses
- Use for complex reasoning tasks

### Test Scenarios
Edit `config/test_scenarios.json` to customize test cases:
- `quick_validation`: Fast basic tests
- `comprehensive_evaluation`: Full evaluation suite
- `performance_stress`: Long context and complex tasks
- `domain_specific`: ML, security, DevOps focused tests

### Ollama Settings
Modify `config/ollama_config.json` for performance tuning:
- GPU memory allocation
- Context length
- Temperature and sampling parameters
- RunPod-specific optimizations

## 🔧 Troubleshooting

### Common Issues

**Ollama not starting:**
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
```

**GPU not detected:**
```bash
nvidia-smi
# If no output, check NVIDIA drivers
```

**Model download fails:**
```bash
# Check disk space
df -h
# Retry download
ollama pull gpt-oss:20b
```

**Out of memory errors:**
- Use gpt-oss:20b instead of 120b
- Reduce context length in config
- Monitor with `nvidia-smi` during tests

### Performance Tips

1. **Optimal Settings for RTX 4090:**
   - Use gpt-oss:20b for best performance
   - Enable mixed precision
   - Set GPU memory fraction to 0.95

2. **RunPod Specific:**
   - Monitor temperature (RunPod instances can get warm)
   - Use persistent storage for model files
   - Consider spot instances for cost savings

3. **Testing Best Practices:**
   - Run one test at a time to avoid memory issues
   - Use GPU monitoring to track resource usage
   - Save results frequently

## 📈 Benchmarking

### Standard Benchmark Suite
```bash
python3 scripts/interactive_test.py --auto-benchmark
```

Includes tests for:
- Code generation and debugging
- Mathematical reasoning
- Creative writing
- Technical analysis
- Logical problem solving

### Custom Benchmarks
Create custom test scenarios in `config/test_scenarios.json` with:
- Custom prompts
- Expected duration estimates
- Reasoning effort levels
- Success criteria

## 🔍 Monitoring Dashboard

The GPU monitor provides real-time metrics:
- GPU utilization %
- VRAM usage
- Temperature
- Power consumption
- System CPU/RAM usage

## 📝 Logging

All tests automatically log:
- Response times
- GPU metrics
- Full model responses
- Error messages
- Performance statistics

Logs are saved in the `logs/` directory with timestamps.

## 🤝 Contributing

To add new test scenarios:
1. Edit `config/test_scenarios.json`
2. Add prompts and expected parameters
3. Test with the interactive tool
4. Document results

## 📄 License

This testing framework is provided as-is for evaluating GPT-OSS models. 

---

## 🆘 Need Help?

1. Check the logs in `logs/` directory
2. Run `nvidia-smi` to verify GPU status
3. Ensure Ollama is running: `ollama list`
4. Monitor resources: `python3 scripts/gpu_monitor.py`

**Happy Testing! 🎉**
