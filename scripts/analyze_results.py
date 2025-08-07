#!/usr/bin/env python3
"""
Results Analysis Script for GPT-OSS Tests
Analyzes test logs and provides performance metrics
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import click

console = Console()

class ResultsAnalyzer:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.results_data = []
        
    def load_results(self):
        """Load all test results from log files"""
        log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
        
        for log_file in log_files:
            try:
                # Try to load as JSON first (from interactive tests)
                with open(log_file, 'r') as f:
                    content = f.read()
                    if content.startswith('{'):
                        data = json.loads(content)
                        self.results_data.append(data)
                    else:
                        # Parse text logs (from bash tests)
                        self._parse_text_log(log_file, content)
            except (json.JSONDecodeError, Exception) as e:
                console.print(f"[yellow]Warning: Could not parse {log_file}: {e}[/yellow]")
                
    def _parse_text_log(self, log_file, content):
        """Parse text-based log files from bash tests"""
        lines = content.split('\n')
        data = {"source": "bash_test"}
        
        for line in lines:
            if line.startswith("Model:"):
                data["model"] = line.split(":", 1)[1].strip()
            elif line.startswith("Timestamp:"):
                data["timestamp"] = line.split(":", 1)[1].strip()
            elif line.startswith("Prompt:"):
                data["prompt"] = line.split(":", 1)[1].strip()
            elif line.startswith("Duration:"):
                duration_str = line.split(":", 1)[1].strip().replace('s', '')
                try:
                    data["duration"] = float(duration_str)
                except ValueError:
                    data["duration"] = 0
        
        # Extract test name from filename
        filename = os.path.basename(log_file)
        if "_" in filename:
            parts = filename.split("_")
            data["test_name"] = parts[0] if len(parts) > 0 else "unknown"
        
        if "model" in data and "duration" in data:
            self.results_data.append(data)
    
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.results_data:
            console.print("[red]No test results found. Run some tests first.[/red]")
            return
        
        df = pd.DataFrame(self.results_data)
        
        # Basic statistics
        summary_table = Table(title="Test Results Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tests", str(len(df)))
        summary_table.add_row("Unique Models", str(df['model'].nunique()) if 'model' in df else "N/A")
        
        if 'duration' in df:
            summary_table.add_row("Avg Response Time", f"{df['duration'].mean():.2f}s")
            summary_table.add_row("Min Response Time", f"{df['duration'].min():.2f}s")
            summary_table.add_row("Max Response Time", f"{df['duration'].max():.2f}s")
        
        console.print(summary_table)
        
        # Model performance comparison
        if 'model' in df and 'duration' in df:
            model_stats = df.groupby('model')['duration'].agg(['count', 'mean', 'std']).round(2)
            
            perf_table = Table(title="Model Performance Comparison")
            perf_table.add_column("Model", style="magenta")
            perf_table.add_column("Tests", style="cyan")
            perf_table.add_column("Avg Time (s)", style="green")
            perf_table.add_column("Std Dev", style="yellow")
            
            for model, stats in model_stats.iterrows():
                perf_table.add_row(
                    model,
                    str(int(stats['count'])),
                    f"{stats['mean']:.2f}",
                    f"{stats['std']:.2f}" if pd.notna(stats['std']) else "N/A"
                )
            
            console.print(perf_table)
    
    def generate_plots(self):
        """Generate performance visualization plots"""
        if not self.results_data:
            console.print("[red]No data to plot.[/red]")
            return
        
        df = pd.DataFrame(self.results_data)
        
        if 'duration' not in df:
            console.print("[yellow]No duration data available for plotting.[/yellow]")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GPT-OSS Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Response time distribution
        axes[0, 0].hist(df['duration'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Response Time Distribution')
        axes[0, 0].set_xlabel('Duration (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Model comparison (if multiple models)
        if 'model' in df and df['model'].nunique() > 1:
            model_data = df.groupby('model')['duration'].mean().sort_values()
            axes[0, 1].bar(range(len(model_data)), model_data.values)
            axes[0, 1].set_title('Average Response Time by Model')
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('Average Duration (seconds)')
            axes[0, 1].set_xticks(range(len(model_data)))
            axes[0, 1].set_xticklabels(model_data.index, rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'Single Model\nNo Comparison Available', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Model Comparison (N/A)')
        
        # Plot 3: Timeline of tests
        if 'timestamp' in df:
            try:
                # Convert timestamps to datetime for plotting
                df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df_time = df.dropna(subset=['datetime']).sort_values('datetime')
                
                if len(df_time) > 1:
                    axes[1, 0].plot(df_time['datetime'], df_time['duration'], 'o-', alpha=0.7)
                    axes[1, 0].set_title('Response Time Over Time')
                    axes[1, 0].set_xlabel('Time')
                    axes[1, 0].set_ylabel('Duration (seconds)')
                    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
                else:
                    axes[1, 0].text(0.5, 0.5, 'Insufficient Data\nfor Timeline', 
                                  ha='center', va='center', transform=axes[1, 0].transAxes)
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'Timeline Data\nNot Available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Response Time Timeline')
        
        # Plot 4: Test type performance (if available)
        if 'test_name' in df and df['test_name'].nunique() > 1:
            test_perf = df.groupby('test_name')['duration'].mean().sort_values()
            axes[1, 1].barh(range(len(test_perf)), test_perf.values)
            axes[1, 1].set_title('Performance by Test Type')
            axes[1, 1].set_xlabel('Average Duration (seconds)')
            axes[1, 1].set_ylabel('Test Type')
            axes[1, 1].set_yticks(range(len(test_perf)))
            axes[1, 1].set_yticklabels(test_perf.index)
        else:
            axes[1, 1].text(0.5, 0.5, 'Test Type Data\nNot Available', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Test Type Performance (N/A)')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"logs/performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        console.print(f"[green]Performance plots saved to: {plot_file}[/green]")
        
        # Show plot if in interactive environment
        try:
            plt.show()
        except Exception:
            console.print("[yellow]Plot saved but cannot display in this environment.[/yellow]")
    
    def export_data(self):
        """Export results data to CSV"""
        if not self.results_data:
            console.print("[red]No data to export.[/red]")
            return
        
        df = pd.DataFrame(self.results_data)
        export_file = f"logs/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(export_file, index=False)
        console.print(f"[green]Data exported to: {export_file}[/green]")

@click.command()
@click.option('--plots', is_flag=True, help='Generate performance plots')
@click.option('--export', is_flag=True, help='Export data to CSV')
@click.option('--log-dir', default='logs', help='Directory containing log files')
def main(plots, export, log_dir):
    """Analyze GPT-OSS test results"""
    
    console.print(Panel.fit(
        "[bold blue]GPT-OSS Results Analyzer[/bold blue]\n"
        "Analyze performance metrics from test runs",
        border_style="blue"
    ))
    
    analyzer = ResultsAnalyzer(log_dir)
    
    console.print(f"[cyan]Loading results from {log_dir}...[/cyan]")
    analyzer.load_results()
    
    if not analyzer.results_data:
        console.print("[red]No test results found. Make sure you've run some tests first.[/red]")
        return
    
    console.print(f"[green]Loaded {len(analyzer.results_data)} test results.[/green]")
    
    # Generate summary
    analyzer.generate_summary()
    
    # Generate plots if requested
    if plots:
        console.print("[cyan]Generating performance plots...[/cyan]")
        analyzer.generate_plots()
    
    # Export data if requested
    if export:
        console.print("[cyan]Exporting data...[/cyan]")
        analyzer.export_data()

if __name__ == "__main__":
    main()
