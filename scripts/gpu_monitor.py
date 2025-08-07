#!/usr/bin/env python3
"""
GPU Monitoring Script for GPT-OSS Testing
Monitors GPU usage during model inference
"""

import time
import json
import psutil
import pynvml
import click
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel

console = Console()

class GPUMonitor:
    def __init__(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = []
            
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
                
        except pynvml.NVMLError as e:
            console.print(f"[red]Error initializing NVIDIA ML: {e}[/red]")
            self.device_count = 0
            self.handles = []
    
    def get_gpu_info(self):
        """Get current GPU information"""
        gpu_info = []
        
        for i, handle in enumerate(self.handles):
            try:
                # GPU name
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024**3  # GB
                mem_total = mem_info.total / 1024**3  # GB
                mem_percent = (mem_info.used / mem_info.total) * 100
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError:
                    temp = 0
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Watts
                except pynvml.NVMLError:
                    power = 0
                
                gpu_info.append({
                    'id': i,
                    'name': name,
                    'memory_used': mem_used,
                    'memory_total': mem_total,
                    'memory_percent': mem_percent,
                    'gpu_utilization': gpu_util,
                    'temperature': temp,
                    'power': power
                })
                
            except pynvml.NVMLError as e:
                console.print(f"[red]Error getting GPU {i} info: {e}[/red]")
                
        return gpu_info
    
    def get_system_info(self):
        """Get system resource information"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used': psutil.virtual_memory().used / 1024**3,  # GB
            'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
        }
    
    def create_monitoring_table(self, gpu_info, system_info):
        """Create a rich table for monitoring display"""
        table = Table(title=f"GPU & System Monitor - {datetime.now().strftime('%H:%M:%S')}")
        
        # GPU columns
        table.add_column("GPU", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        table.add_column("GPU %", style="green")
        table.add_column("Memory", style="yellow")
        table.add_column("Temp °C", style="red")
        table.add_column("Power W", style="blue")
        
        for gpu in gpu_info:
            memory_str = f"{gpu['memory_used']:.1f}/{gpu['memory_total']:.1f}GB ({gpu['memory_percent']:.1f}%)"
            
            table.add_row(
                str(gpu['id']),
                gpu['name'][:20],  # Truncate long names
                f"{gpu['gpu_utilization']}%",
                memory_str,
                f"{gpu['temperature']}",
                f"{gpu['power']:.1f}"
            )
        
        # Add system info row
        system_memory = f"{system_info['memory_used']:.1f}/{system_info['memory_total']:.1f}GB ({system_info['memory_percent']:.1f}%)"
        table.add_row(
            "SYS",
            "System CPU/RAM",
            f"{system_info['cpu_percent']:.1f}%",
            system_memory,
            "-",
            "-"
        )
        
        return table
    
    def log_metrics(self, gpu_info, system_info, log_file):
        """Log metrics to file"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'gpu_info': gpu_info,
            'system_info': system_info
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

@click.command()
@click.option('--interval', default=2, help='Update interval in seconds')
@click.option('--log-file', help='Log metrics to file')
@click.option('--duration', help='Monitoring duration in seconds (default: infinite)')
def main(interval, log_file, duration):
    """Monitor GPU usage during GPT-OSS testing"""
    
    console.print(Panel.fit(
        "[bold blue]GPU Monitor for GPT-OSS Testing[/bold blue]\n"
        "Real-time monitoring of GPU and system resources",
        border_style="blue"
    ))
    
    monitor = GPUMonitor()
    
    if monitor.device_count == 0:
        console.print("[red]No NVIDIA GPUs found or NVIDIA ML library not available.[/red]")
        return
    
    console.print(f"[green]Found {monitor.device_count} GPU(s). Starting monitor...[/green]")
    console.print("[yellow]Press Ctrl+C to stop monitoring[/yellow]")
    
    start_time = time.time()
    duration_seconds = float(duration) if duration else float('inf')
    
    try:
        with Live(console=console, refresh_per_second=1/interval) as live:
            while True:
                current_time = time.time()
                if current_time - start_time > duration_seconds:
                    break
                
                # Get current metrics
                gpu_info = monitor.get_gpu_info()
                system_info = monitor.get_system_info()
                
                # Create and update display
                table = monitor.create_monitoring_table(gpu_info, system_info)
                live.update(table)
                
                # Log if requested
                if log_file:
                    monitor.log_metrics(gpu_info, system_info, log_file)
                
                time.sleep(interval)
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user.[/yellow]")
    
    if log_file:
        console.print(f"[green]Metrics logged to: {log_file}[/green]")

if __name__ == "__main__":
    main()
