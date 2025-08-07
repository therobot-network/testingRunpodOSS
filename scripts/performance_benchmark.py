#!/usr/bin/env python3
"""
Advanced Performance Benchmarking Script for GPT-OSS
Measures TTFT, tokens/second, memory usage, and other performance metrics using real prompts
"""

import os
import csv
import time
import json
import random
import subprocess
import threading
import pynvml
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.live import Live
import click

console = Console()

class PerformanceBenchmark:
    def __init__(self, model="gpt-oss:20b", data_file="data/101-200.csv"):
        self.model = model
        self.data_file = data_file
        self.results = []
        self.prompts = []
        self.log_dir = "logs"
        
        # Initialize NVIDIA ML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
        except:
            self.gpu_available = False
            console.print("[yellow]GPU monitoring not available[/yellow]")
        
        # Ensure logs directory exists
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_prompts(self, num_prompts=None):
        """Load prompts from CSV file"""
        console.print(f"[cyan]Loading prompts from {self.data_file}...[/cyan]")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'full_prompt' in row:
                        self.prompts.append({
                            'question': row.get('User Search Question', ''),
                            'prompt': row['full_prompt']
                        })
        except FileNotFoundError:
            console.print(f"[red]Error: Could not find {self.data_file}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Error loading prompts: {e}[/red]")
            return False
        
        if num_prompts:
            self.prompts = random.sample(self.prompts, min(num_prompts, len(self.prompts)))
        
        console.print(f"[green]Loaded {len(self.prompts)} prompts[/green]")
        return True
    
    def get_gpu_metrics(self):
        """Get current GPU metrics"""
        if not self.gpu_available:
            return None
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return {
                'memory_used_mb': mem_info.used / 1024**2,
                'memory_total_mb': mem_info.total / 1024**2,
                'memory_percent': (mem_info.used / mem_info.total) * 100,
                'gpu_utilization': util.gpu,
                'temperature': temp
            }
        except:
            return None
    
    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) / 4
    
    def run_inference(self, prompt, timeout=300):
        """Run inference and measure detailed performance metrics"""
        # Get initial GPU state
        gpu_before = self.get_gpu_metrics()
        
        # Prepare the command
        cmd = ['ollama', 'run', self.model, prompt]
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run the inference
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Track first token time and collect output
            output_lines = []
            first_token_time = None
            
            for line in iter(process.stdout.readline, ''):
                if line.strip():  # Non-empty line
                    if first_token_time is None:
                        first_token_time = time.time()
                    output_lines.append(line.rstrip())
            
            process.wait(timeout=timeout)
            end_time = time.time()
            
            # Combine all output
            full_response = '\n'.join(output_lines)
            
            # Calculate metrics
            total_time = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else total_time
            
            # Token estimation
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(full_response)
            total_tokens = input_tokens + output_tokens
            
            # Calculate tokens per second
            if total_time > ttft:
                tokens_per_second = output_tokens / (total_time - ttft)
            else:
                tokens_per_second = output_tokens / total_time if total_time > 0 else 0
            
            # Get final GPU state
            gpu_after = self.get_gpu_metrics()
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'prompt_length': len(prompt),
                'response_length': len(full_response),
                'input_tokens_est': int(input_tokens),
                'output_tokens_est': int(output_tokens),
                'total_tokens_est': int(total_tokens),
                'total_time_seconds': round(total_time, 3),
                'ttft_seconds': round(ttft, 3),
                'tokens_per_second': round(tokens_per_second, 2),
                'throughput_tokens_per_second': round(total_tokens / total_time, 2) if total_time > 0 else 0,
                'gpu_before': gpu_before,
                'gpu_after': gpu_after,
                'response': full_response[:500] + "..." if len(full_response) > 500 else full_response,
                'success': True
            }
            
            return result
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'error': 'Timeout',
                'success': False
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'error': str(e),
                'success': False
            }
    
    def run_benchmark(self, num_tests=10, save_results=True):
        """Run the complete benchmark suite"""
        if not self.prompts:
            console.print("[red]No prompts loaded. Run load_prompts() first.[/red]")
            return
        
        console.print(Panel.fit(
            f"[bold blue]GPT-OSS Performance Benchmark[/bold blue]\n"
            f"Model: {self.model}\n"
            f"Tests: {min(num_tests, len(self.prompts))}\n"
            f"Measuring: TTFT, Tokens/sec, GPU usage",
            border_style="blue"
        ))
        
        # Select prompts for testing
        test_prompts = random.sample(self.prompts, min(num_tests, len(self.prompts)))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            
            task = progress.add_task("Running benchmark tests...", total=len(test_prompts))
            
            for i, prompt_data in enumerate(test_prompts, 1):
                prompt = prompt_data['prompt']
                question = prompt_data['question']
                
                progress.update(task, description=f"Test {i}/{len(test_prompts)}: {question[:50]}...")
                
                console.print(f"\n[yellow]Test {i}: {question}[/yellow]")
                console.print(f"[dim]Prompt length: {len(prompt)} chars[/dim]")
                
                result = self.run_inference(prompt)
                
                if result['success']:
                    console.print(f"[green]✅ Completed in {result['total_time_seconds']}s[/green]")
                    console.print(f"[cyan]   TTFT: {result['ttft_seconds']}s | Tokens/sec: {result['tokens_per_second']} | Output: {result['output_tokens_est']} tokens[/cyan]")
                    
                    if result.get('gpu_after'):
                        gpu = result['gpu_after']
                        console.print(f"[blue]   GPU: {gpu['memory_percent']:.1f}% VRAM | {gpu['gpu_utilization']}% util | {gpu['temperature']}°C[/blue]")
                else:
                    console.print(f"[red]❌ Failed: {result.get('error', 'Unknown error')}[/red]")
                
                result['test_number'] = i
                result['question'] = question
                self.results.append(result)
                
                progress.advance(task)
        
        # Save results if requested
        if save_results:
            self.save_results()
        
        # Display summary
        self.display_summary()
    
    def save_results(self):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = f"{self.log_dir}/performance_benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV summary
        csv_file = f"{self.log_dir}/performance_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=[
                    'test_number', 'question', 'total_time_seconds', 'ttft_seconds', 
                    'tokens_per_second', 'throughput_tokens_per_second', 
                    'input_tokens_est', 'output_tokens_est', 'success'
                ])
                writer.writeheader()
                for result in self.results:
                    if result['success']:
                        writer.writerow({
                            'test_number': result.get('test_number', 0),
                            'question': result.get('question', '')[:100],
                            'total_time_seconds': result['total_time_seconds'],
                            'ttft_seconds': result['ttft_seconds'],
                            'tokens_per_second': result['tokens_per_second'],
                            'throughput_tokens_per_second': result['throughput_tokens_per_second'],
                            'input_tokens_est': result['input_tokens_est'],
                            'output_tokens_est': result['output_tokens_est'],
                            'success': result['success']
                        })
        
        console.print(f"[green]Results saved to:[/green]")
        console.print(f"  📄 {json_file}")
        console.print(f"  📊 {csv_file}")
    
    def display_summary(self):
        """Display benchmark results summary"""
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            console.print("[red]No successful tests to summarize[/red]")
            return
        
        # Calculate statistics
        total_times = [r['total_time_seconds'] for r in successful_results]
        ttfts = [r['ttft_seconds'] for r in successful_results]
        tokens_per_sec = [r['tokens_per_second'] for r in successful_results]
        throughputs = [r['throughput_tokens_per_second'] for r in successful_results]
        
        # Create summary table
        table = Table(title="Performance Benchmark Summary")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Min", style="green")
        table.add_column("Max", style="red")
        table.add_column("Average", style="yellow")
        table.add_column("Median", style="blue")
        
        def stats(values):
            if not values:
                return "N/A", "N/A", "N/A", "N/A"
            sorted_vals = sorted(values)
            return (
                f"{min(values):.2f}",
                f"{max(values):.2f}",
                f"{sum(values)/len(values):.2f}",
                f"{sorted_vals[len(sorted_vals)//2]:.2f}"
            )
        
        table.add_row("Total Time (s)", *stats(total_times))
        table.add_row("TTFT (s)", *stats(ttfts))
        table.add_row("Tokens/sec (output)", *stats(tokens_per_sec))
        table.add_row("Throughput (total)", *stats(throughputs))
        
        console.print(table)
        
        # Additional metrics
        console.print(f"\n[bold]Test Results:[/bold]")
        console.print(f"✅ Successful tests: {len(successful_results)}/{len(self.results)}")
        console.print(f"📝 Average prompt length: {sum(r['prompt_length'] for r in successful_results)/len(successful_results):.0f} chars")
        console.print(f"📤 Average response length: {sum(r['response_length'] for r in successful_results)/len(successful_results):.0f} chars")
        console.print(f"🎯 Average output tokens: {sum(r['output_tokens_est'] for r in successful_results)/len(successful_results):.0f}")

@click.command()
@click.option('--model', default='gpt-oss:20b', help='Model to benchmark')
@click.option('--data-file', default='data/101-200.csv', help='CSV file with prompts')
@click.option('--num-prompts', default=10, help='Number of prompts to test')
@click.option('--save/--no-save', default=True, help='Save results to files')
def main(model, data_file, num_prompts, save):
    """Advanced GPT-OSS Performance Benchmark
    
    Measures TTFT (Time to First Token), tokens per second, GPU usage,
    and other performance metrics using real RAG prompts.
    """
    
    benchmark = PerformanceBenchmark(model=model, data_file=data_file)
    
    # Load prompts
    if not benchmark.load_prompts(num_prompts):
        return
    
    # Run benchmark
    benchmark.run_benchmark(num_tests=num_prompts, save_results=save)

if __name__ == "__main__":
    main()
