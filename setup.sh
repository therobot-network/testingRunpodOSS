#!/bin/bash

# Ollama GPT-OSS Testing Setup Script for RunPod 4090 GPU
# This script sets up the environment for testing OpenAI's GPT-OSS models via Ollama

set -e

echo "🚀 Setting up Ollama GPT-OSS Testing Environment for RunPod 4090..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🛠️  Installing essential packages..."
sudo apt install -y curl wget git python3 python3-pip htop nvtop jq

# Install Ollama
echo "🦙 Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
echo "🔄 Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
sleep 10

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Pull GPT-OSS models (starting with 20B for testing)
echo "📥 Pulling GPT-OSS 20B model..."
ollama pull gpt-oss:20b

# Optional: Pull 120B model if you have enough VRAM (requires ~65GB)
# Uncomment the line below if you want to test the larger model
# echo "📥 Pulling GPT-OSS 120B model..."
# ollama pull gpt-oss:120b

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip3 install -r requirements.txt

# Make test scripts executable
echo "🔧 Making test scripts executable..."
chmod +x tests/*.sh
chmod +x scripts/*.py

# Create logs directory
mkdir -p logs

echo "✅ Setup complete! You can now run tests with:"
echo "   ./run_basic_tests.sh"
echo "   python3 scripts/interactive_test.py"
echo ""
echo "📊 Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "🦙 Check Ollama status with: ollama list"
