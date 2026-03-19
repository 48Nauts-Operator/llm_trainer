#!/usr/bin/env python3
"""
LLM Trainer - Model Testing Script
Tests trained models for accuracy and performance.
"""

import argparse
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress
import mlx.core as mx
from mlx_lm import load, generate

console = Console()

class ModelTester:
    """Test and evaluate trained models"""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        
        # Load model info if available
        self.model_info = self._load_model_info()
        
    def _load_model_info(self) -> Dict[str, Any]:
        """Load model information from training"""
        info_file = self.model_path / "model_info.json"
        if info_file.exists():
            with open(info_file) as f:
                return json.load(f)
        return {}
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        console.print("[blue]Loading trained model...[/blue]")
        
        # Determine base model
        if self.base_model:
            base_model = self.base_model
        elif "base_model" in self.model_info:
            base_model = self.model_info["base_model"]
        else:
            base_model = "Qwen/Qwen2.5-3B-Instruct"  # Default
        
        adapter_path = self.model_path / "adapters"
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapters found at {adapter_path}")
        
        try:
            self.model, self.tokenizer = load(base_model, adapter_path=str(adapter_path))
            console.print(f"[green]✅ Model loaded: {base_model}[/green]")
            console.print(f"[green]✅ Adapters loaded: {adapter_path}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise
    
    def classify_query(self, query: str, temperature: float = 0.1, max_tokens: int = 100) -> Dict[str, Any]:
        """Classify a single query"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=query,
                max_tokens=max_tokens,
                temp=temperature
            )
            
            response_time = time.time() - start_time
            
            # Try to parse JSON response
            try:
                parsed_response = json.loads(response.strip())
            except json.JSONDecodeError:
                # If not JSON, return raw response
                parsed_response = {"raw_response": response.strip()}
            
            return {
                "query": query,
                "response": parsed_response,
                "raw_response": response.strip(),
                "response_time": response_time,
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "response_time": time.time() - start_time,
                "success": False
            }
    
    def test_with_test_data(self, test_data_path: str) -> Dict[str, Any]:
        """Test model against test dataset"""
        console.print(f"[blue]Testing with dataset: {test_data_path}[/blue]")
        
        test_data = []
        with open(test_data_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        results = []
        correct = 0
        total = len(test_data)
        total_time = 0
        
        for item in track(test_data, description="Testing"):
            result = self.classify_query(item["prompt"])
            results.append(result)
            total_time += result["response_time"]
            
            if result["success"]:
                # Compare with expected result
                expected = json.loads(item["completion"])
                if result["response"] == expected:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        avg_response_time = total_time / total if total > 0 else 0
        
        return {
            "total_tests": total,
            "correct": correct,
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "total_time": total_time,
            "results": results
        }
    
    def test_interactive_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Test with provided queries"""
        console.print("[blue]Testing with interactive queries...[/blue]")
        
        results = []
        for query in track(queries, description="Testing queries"):
            result = self.classify_query(query)
            results.append(result)
        
        return results
    
    def benchmark_performance(self, num_queries: int = 100) -> Dict[str, float]:
        """Benchmark model performance"""
        console.print(f"[blue]Running performance benchmark ({num_queries} queries)...[/blue]")
        
        # Use a simple test query
        test_query = "Classify this business email: I'm interested in your services"
        
        times = []
        for _ in track(range(num_queries), description="Benchmarking"):
            start = time.time()
            _ = self.classify_query(test_query)
            times.append(time.time() - start)
        
        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "queries_per_second": num_queries / sum(times)
        }
    
    def generate_report(self, test_results: Dict[str, Any], benchmark_results: Dict[str, float] = None) -> str:
        """Generate a comprehensive test report"""
        report = []
        
        # Header
        report.append("# Model Test Report")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model info
        if self.model_info:
            report.append("## Model Information")
            report.append(f"- **Name**: {self.model_info.get('name', 'Unknown')}")
            report.append(f"- **Base Model**: {self.model_info.get('base_model', 'Unknown')}")
            report.append(f"- **Training Date**: {self.model_info.get('training_date', 'Unknown')}")
            
            if "training_parameters" in self.model_info:
                params = self.model_info["training_parameters"]
                report.append(f"- **Training Iterations**: {params.get('iterations', 'Unknown')}")
                report.append(f"- **Batch Size**: {params.get('batch_size', 'Unknown')}")
                report.append(f"- **Learning Rate**: {params.get('learning_rate', 'Unknown')}")
            report.append("")
        
        # Test results
        report.append("## Test Results")
        report.append(f"- **Total Tests**: {test_results['total_tests']}")
        report.append(f"- **Correct**: {test_results['correct']}")
        report.append(f"- **Accuracy**: {test_results['accuracy']:.1%}")
        report.append(f"- **Average Response Time**: {test_results['avg_response_time']:.3f}s")
        report.append("")
        
        # Performance benchmark
        if benchmark_results:
            report.append("## Performance Benchmark")
            report.append(f"- **Average Time**: {benchmark_results['avg_time']:.3f}s")
            report.append(f"- **Min Time**: {benchmark_results['min_time']:.3f}s")
            report.append(f"- **Max Time**: {benchmark_results['max_time']:.3f}s")
            report.append(f"- **Queries/Second**: {benchmark_results['queries_per_second']:.1f}")
            report.append("")
        
        # Detailed results (first 10 examples)
        report.append("## Sample Test Cases")
        for i, result in enumerate(test_results['results'][:10]):
            report.append(f"### Test Case {i+1}")
            report.append(f"- **Query**: {result['query']}")
            if result['success']:
                response_json = json.dumps(result['response'], indent=2)
                report.append(f"- **Response**: ```json\n{response_json}\n```")
                report.append(f"- **Response Time**: {result['response_time']:.3f}s")
            else:
                report.append(f"- **Error**: {result.get('error', 'Unknown error')}")
            report.append("")
        
        return "\n".join(report)

def create_default_test_queries() -> List[str]:
    """Create default test queries for interactive testing"""
    return [
        "Classify this business email: I'm interested in your consulting services",
        "Classify this business email: The system isn't working properly",
        "Classify this business email: Can we schedule a meeting next week?",
        "Classify this business email: I need help with the implementation",
        "Classify this business email: What are your rates for strategic planning?",
        "Classify this business email: The server is down and we can't access the system",
        "Classify this business email: I'd like to set up a consultation",
        "Classify this business email: We need technical support urgently",
    ]

def main():
    parser = argparse.ArgumentParser(description="Test trained AI models")
    parser.add_argument("--model-path", required=True, help="Path to trained model directory")
    parser.add_argument("--base-model", help="Base model name (if not in model info)")
    parser.add_argument("--test-data", help="Path to test data file (JSONL format)")
    parser.add_argument("--queries", nargs="+", help="Custom queries to test")
    parser.add_argument("--benchmark", type=int, default=0, help="Run performance benchmark (number of queries)")
    parser.add_argument("--output", help="Output file for detailed report")
    parser.add_argument("--interactive", action="store_true", help="Interactive testing mode")
    
    args = parser.parse_args()
    
    console.print("[bold green]🧪 LLM Trainer - Model Testing[/bold green]")
    
    try:
        # Initialize tester
        tester = ModelTester(args.model_path, args.base_model)
        tester.load_model()
        
        test_results = None
        benchmark_results = None
        
        # Test with test data
        if args.test_data:
            if not Path(args.test_data).exists():
                console.print(f"[red]❌ Test data file not found: {args.test_data}[/red]")
                return 1
            test_results = tester.test_with_test_data(args.test_data)
        
        # Test with custom queries or defaults
        elif args.queries:
            results = tester.test_interactive_queries(args.queries)
            test_results = {
                "total_tests": len(results),
                "results": results,
                "correct": sum(1 for r in results if r["success"]),
                "accuracy": sum(1 for r in results if r["success"]) / len(results),
                "avg_response_time": sum(r["response_time"] for r in results) / len(results),
                "total_time": sum(r["response_time"] for r in results)
            }
        
        elif args.interactive:
            console.print("[yellow]Interactive mode - enter queries (Ctrl+C to exit):[/yellow]")
            while True:
                try:
                    query = console.input("\nQuery: ")
                    if not query.strip():
                        continue
                    
                    result = tester.classify_query(query)
                    
                    if result["success"]:
                        console.print(f"[green]Response:[/green] {json.dumps(result['response'], indent=2)}")
                        console.print(f"[blue]Time:[/blue] {result['response_time']:.3f}s")
                    else:
                        console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
                        
                except KeyboardInterrupt:
                    console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                    break
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            
            return 0
        
        else:
            # Use default queries
            default_queries = create_default_test_queries()
            results = tester.test_interactive_queries(default_queries)
            test_results = {
                "total_tests": len(results),
                "results": results,
                "correct": sum(1 for r in results if r["success"]),
                "accuracy": sum(1 for r in results if r["success"]) / len(results),
                "avg_response_time": sum(r["response_time"] for r in results) / len(results),
                "total_time": sum(r["response_time"] for r in results)
            }
        
        # Run benchmark if requested
        if args.benchmark > 0:
            benchmark_results = tester.benchmark_performance(args.benchmark)
        
        # Display results
        if test_results:
            table = Table(title="Test Results Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total Tests", str(test_results["total_tests"]))
            table.add_row("Successful", str(test_results["correct"]))
            table.add_row("Accuracy", f"{test_results['accuracy']:.1%}")
            table.add_row("Avg Response Time", f"{test_results['avg_response_time']:.3f}s")
            
            if benchmark_results:
                table.add_row("Benchmark Queries", str(args.benchmark))
                table.add_row("Queries/Second", f"{benchmark_results['queries_per_second']:.1f}")
            
            console.print(table)
        
        # Generate report if requested
        if args.output and test_results:
            report = tester.generate_report(test_results, benchmark_results)
            with open(args.output, 'w') as f:
                f.write(report)
            console.print(f"[green]📝 Detailed report saved: {args.output}[/green]")
        
        # Show sample results
        if test_results and test_results["results"]:
            console.print("\n[bold]Sample Results:[/bold]")
            for i, result in enumerate(test_results["results"][:3]):
                console.print(f"\n[cyan]Query {i+1}:[/cyan] {result['query']}")
                if result["success"]:
                    console.print(f"[green]Response:[/green] {json.dumps(result['response'], indent=2)}")
                else:
                    console.print(f"[red]Error:[/red] {result.get('error', 'Unknown')}")
        
        console.print(f"\n[bold green]✅ Testing complete![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())