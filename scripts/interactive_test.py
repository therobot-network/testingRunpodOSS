#!/usr/bin/env python3
"""
Interactive GPT-OSS Testing Script
Provides an interactive CLI for testing GPT-OSS models with various configurations
"""

import os
import time
import json
import click
import subprocess
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

class OllamaGPTOSSTest:
    def __init__(self):
        self.available_models = []
        self.current_model = None
        self.test_history = []
        self.log_dir = "logs"
        
        # Ensure logs directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
    def check_ollama_status(self):
        """Check if Ollama is running and get available models"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            self.available_models = []
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    if 'gpt-oss' in model_name:
                        self.available_models.append(model_name)
                        
            return True
        except subprocess.CalledProcessError:
            return False
    
    def display_models(self):
        """Display available GPT-OSS models"""
        if not self.available_models:
            console.print("[red]No GPT-OSS models found. Please run setup.sh first.[/red]")
            return
            
        table = Table(title="Available GPT-OSS Models")
        table.add_column("Index", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Size", style="green")
        
        for i, model in enumerate(self.available_models):
            size = "~14GB" if "20b" in model else "~65GB" if "120b" in model else "Unknown"
            table.add_row(str(i + 1), model, size)
            
        console.print(table)
    
    def select_model(self):
        """Let user select a model"""
        if not self.available_models:
            console.print("[red]No models available.[/red]")
            return False
            
        self.display_models()
        
        while True:
            try:
                choice = Prompt.ask("Select model number", default="1")
                index = int(choice) - 1
                
                if 0 <= index < len(self.available_models):
                    self.current_model = self.available_models[index]
                    console.print(f"[green]Selected: {self.current_model}[/green]")
                    return True
                else:
                    console.print("[red]Invalid selection. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number.[/red]")
    
    def run_test(self, prompt, reasoning_effort="medium"):
        """Run a test with the selected model"""
        if not self.current_model:
            console.print("[red]No model selected.[/red]")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.log_dir}/interactive_test_{timestamp}.log"
        
        # Prepare the full prompt with reasoning effort
        full_prompt = f"[Reasoning effort: {reasoning_effort}] {prompt}"
        
        console.print(f"[blue]Running test with {self.current_model}...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    ['ollama', 'run', self.current_model, full_prompt],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Log the test
                test_data = {
                    "timestamp": timestamp,
                    "model": self.current_model,
                    "prompt": prompt,
                    "reasoning_effort": reasoning_effort,
                    "response": result.stdout,
                    "duration": duration
                }
                
                with open(log_file, 'w') as f:
                    json.dump(test_data, f, indent=2)
                
                self.test_history.append(test_data)
                
                progress.remove_task(task)
                
                # Display results
                console.print(Panel(
                    result.stdout,
                    title=f"Response ({duration:.2f}s)",
                    border_style="green"
                ))
                
                return test_data
                
            except subprocess.CalledProcessError as e:
                progress.remove_task(task)
                console.print(f"[red]Error running test: {e}[/red]")
                return None
    
    def benchmark_mode(self):
        """Run predefined benchmark tests"""
        benchmarks = [
            {
                "name": "Code Generation",
                "prompt": "Write a Python class for a binary search tree with insert, search, and delete methods. Include proper error handling.",
                "effort": "high"
            },
            {
                "name": "Mathematical Reasoning",
                "prompt": "Solve this step by step: If a train travels 240 miles in 3 hours, and then increases its speed by 20 mph for the next 2 hours, how far did it travel in total?",
                "effort": "medium"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short story (200 words) about an AI discovering emotions for the first time.",
                "effort": "low"
            },
            {
                "name": "Technical Analysis",
                "prompt": "Explain the differences between REST and GraphQL APIs, including pros and cons of each approach.",
                "effort": "medium"
            }
        ]
        
        if not self.current_model:
            console.print("[red]Please select a model first.[/red]")
            return
        
        console.print(f"[yellow]Running benchmark suite with {self.current_model}[/yellow]")
        
        results = []
        for benchmark in benchmarks:
            console.print(f"\n[cyan]Running: {benchmark['name']}[/cyan]")
            result = self.run_test(benchmark['prompt'], benchmark['effort'])
            if result:
                results.append(result)
        
        # Summary
        if results:
            total_time = sum(r['duration'] for r in results)
            avg_time = total_time / len(results)
            
            summary_table = Table(title="Benchmark Results Summary")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Tests Completed", str(len(results)))
            summary_table.add_row("Total Time", f"{total_time:.2f}s")
            summary_table.add_row("Average Time", f"{avg_time:.2f}s")
            summary_table.add_row("Model Used", self.current_model)
            
            console.print(summary_table)

@click.command()
@click.option('--auto-benchmark', is_flag=True, help='Run automatic benchmark suite')
@click.option('--model', help='Specify model to use (gpt-oss:20b or gpt-oss:120b)')
def main(auto_benchmark, model):
    """Interactive GPT-OSS Testing Tool"""
    
    console.print(Panel.fit(
        "[bold blue]GPT-OSS Interactive Testing Tool[/bold blue]\n"
        "Test OpenAI's open-source models via Ollama",
        border_style="blue"
    ))
    
    tester = OllamaGPTOSSTest()
    
    # Check Ollama status
    if not tester.check_ollama_status():
        console.print("[red]Ollama is not running or not installed. Please run setup.sh first.[/red]")
        return
    
    # Auto-select model if specified
    if model:
        if model in tester.available_models:
            tester.current_model = model
            console.print(f"[green]Using specified model: {model}[/green]")
        else:
            console.print(f"[red]Model {model} not found. Available models:[/red]")
            tester.display_models()
            return
    
    # Run auto benchmark if requested
    if auto_benchmark:
        if not tester.current_model:
            tester.select_model()
        tester.benchmark_mode()
        return
    
    # Interactive mode
    while True:
        if not tester.current_model:
            if not tester.select_model():
                break
        
        console.print("\n[yellow]Options:[/yellow]")
        console.print("1. Run custom test")
        console.print("2. Run benchmark suite")
        console.print("3. Change model")
        console.print("4. View test history")
        console.print("5. Exit")
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="1")
        
        if choice == "1":
            prompt = Prompt.ask("Enter your test prompt")
            effort = Prompt.ask("Reasoning effort", choices=["low", "medium", "high"], default="medium")
            tester.run_test(prompt, effort)
            
        elif choice == "2":
            tester.benchmark_mode()
            
        elif choice == "3":
            tester.select_model()
            
        elif choice == "4":
            if tester.test_history:
                for i, test in enumerate(tester.test_history[-5:], 1):  # Show last 5
                    console.print(f"{i}. {test['timestamp']} - {test['duration']:.2f}s - {test['model']}")
            else:
                console.print("[yellow]No test history available.[/yellow]")
                
        elif choice == "5":
            console.print("[green]Goodbye![/green]")
            break

if __name__ == "__main__":
    main()
