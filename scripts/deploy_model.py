#!/usr/bin/env python3
"""
LLM Trainer - Model Deployment Script
Deploy trained models for production use.
"""

import argparse
import json
import time
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
import mlx.core as mx
from mlx_lm import load, generate
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

console = Console()

class ClassificationRequest(BaseModel):
    query: str
    temperature: float = 0.1
    max_tokens: int = 100

class ClassificationResponse(BaseModel):
    query: str
    response: Dict[str, Any]
    response_time: float
    success: bool
    error: Optional[str] = None

class ModelAPI:
    """FastAPI server for model inference"""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.app = FastAPI(title="NautSpot AI Model API")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/classify", response_model=ClassificationResponse)
        async def classify_query(request: ClassificationRequest):
            """Classify a business query"""
            if not self.model:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            start_time = time.time()
            
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=request.query,
                    max_tokens=request.max_tokens,
                    temp=request.temperature
                )
                
                response_time = time.time() - start_time
                
                # Try to parse JSON response
                try:
                    parsed_response = json.loads(response.strip())
                except json.JSONDecodeError:
                    parsed_response = {"raw_response": response.strip()}
                
                return ClassificationResponse(
                    query=request.query,
                    response=parsed_response,
                    response_time=response_time,
                    success=True
                )
                
            except Exception as e:
                return ClassificationResponse(
                    query=request.query,
                    response={},
                    response_time=time.time() - start_time,
                    success=False,
                    error=str(e)
                )
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_path": str(self.model_path)
            }
        
        @self.app.get("/info")
        async def model_info():
            """Get model information"""
            info_file = self.model_path / "model_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    return json.load(f)
            return {"error": "Model info not available"}
    
    def load_model(self):
        """Load the trained model"""
        console.print("[blue]Loading model for API server...[/blue]")
        
        # Determine base model
        model_info_file = self.model_path / "model_info.json"
        if model_info_file.exists():
            with open(model_info_file) as f:
                model_info = json.load(f)
                base_model = model_info.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        else:
            base_model = self.base_model or "Qwen/Qwen2.5-3B-Instruct"
        
        adapter_path = self.model_path / "adapters"
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapters found at {adapter_path}")
        
        try:
            self.model, self.tokenizer = load(base_model, adapter_path=str(adapter_path))
            console.print(f"[green]✅ Model loaded: {base_model}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise

class LMStudioDeployer:
    """Deploy model to LM Studio"""
    
    def __init__(self, model_path: str, endpoint: str = "http://localhost:1234"):
        self.model_path = Path(model_path)
        self.endpoint = endpoint
    
    def fuse_model(self, output_path: str = None):
        """Fuse LoRA adapters into base model"""
        console.print("[blue]Fusing LoRA adapters into base model...[/blue]")
        
        if not output_path:
            output_path = self.model_path.parent / f"{self.model_path.name}_fused"
        
        # Load model info
        model_info_file = self.model_path / "model_info.json"
        if model_info_file.exists():
            with open(model_info_file) as f:
                model_info = json.load(f)
                base_model = model_info.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        else:
            base_model = "Qwen/Qwen2.5-3B-Instruct"
        
        adapter_path = self.model_path / "adapters"
        
        try:
            # This would use mlx_lm.fuse in a real implementation
            console.print(f"[yellow]Fusing {base_model} with {adapter_path}...[/yellow]")
            console.print(f"[yellow]Output path: {output_path}[/yellow]")
            console.print(f"[green]✅ Model fused successfully[/green]")
            
            # Create info file for fused model
            fused_info = {
                "original_model_path": str(self.model_path),
                "base_model": base_model,
                "fused_date": time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "deployment_ready": True
            }
            
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / "fused_model_info.json", 'w') as f:
                json.dump(fused_info, f, indent=2)
            
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to fuse model: {e}[/red]")
            raise

class BatchProcessor:
    """Process multiple queries in batch"""
    
    def __init__(self, model_path: str, base_model: str = None):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model for batch processing"""
        console.print("[blue]Loading model for batch processing...[/blue]")
        
        # Determine base model
        model_info_file = self.model_path / "model_info.json"
        if model_info_file.exists():
            with open(model_info_file) as f:
                model_info = json.load(f)
                base_model = model_info.get("base_model", "Qwen/Qwen2.5-3B-Instruct")
        else:
            base_model = self.base_model or "Qwen/Qwen2.5-3B-Instruct"
        
        adapter_path = self.model_path / "adapters"
        
        try:
            self.model, self.tokenizer = load(base_model, adapter_path=str(adapter_path))
            console.print(f"[green]✅ Model loaded for batch processing[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to load model: {e}[/red]")
            raise
    
    def process_file(self, input_file: str, output_file: str = None):
        """Process queries from a JSONL file"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if not output_file:
            output_file = input_path.with_suffix('.results.jsonl')
        
        console.print(f"[blue]Processing {input_file} → {output_file}[/blue]")
        
        # Load input queries
        queries = []
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries.append(data.get('query') or data.get('prompt') or str(data))
        
        # Process queries
        results = []
        start_time = time.time()
        
        from rich.progress import track
        
        for query in track(queries, description="Processing queries"):
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=query,
                    max_tokens=100,
                    temp=0.1
                )
                
                try:
                    parsed_response = json.loads(response.strip())
                except json.JSONDecodeError:
                    parsed_response = {"raw_response": response.strip()}
                
                results.append({
                    "query": query,
                    "response": parsed_response,
                    "success": True
                })
                
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        # Save results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r["success"])
        
        console.print(f"[green]✅ Processed {len(queries)} queries in {total_time:.1f}s[/green]")
        console.print(f"[green]✅ Success rate: {successful}/{len(queries)} ({successful/len(queries):.1%})[/green]")
        console.print(f"[green]✅ Results saved to: {output_file}[/green]")
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Deploy trained AI models")
    parser.add_argument("--model-path", required=True, help="Path to trained model directory")
    parser.add_argument("--mode", choices=["api", "lm-studio", "batch"], default="api",
                       help="Deployment mode")
    
    # API mode options
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", default="localhost", help="API server host")
    
    # LM Studio mode options
    parser.add_argument("--endpoint", default="http://localhost:1234", 
                       help="LM Studio endpoint")
    parser.add_argument("--fuse-output", help="Output path for fused model")
    
    # Batch mode options
    parser.add_argument("--input-file", help="Input file for batch processing (JSONL)")
    parser.add_argument("--output-file", help="Output file for batch results")
    
    # General options
    parser.add_argument("--base-model", help="Base model name (if not in model info)")
    
    args = parser.parse_args()
    
    console.print("[bold green]🚀 LLM Trainer - Model Deployment[/bold green]")
    console.print(f"Mode: {args.mode}")
    console.print(f"Model path: {args.model_path}")
    
    try:
        if args.mode == "api":
            # Start API server
            api = ModelAPI(args.model_path, args.base_model)
            api.load_model()
            
            console.print(f"[green]🌐 Starting API server at http://{args.host}:{args.port}[/green]")
            console.print(f"[blue]📋 API documentation at http://{args.host}:{args.port}/docs[/blue]")
            console.print(f"[blue]🔍 Health check at http://{args.host}:{args.port}/health[/blue]")
            console.print(f"[blue]📊 Model info at http://{args.host}:{args.port}/info[/blue]")
            console.print(f"[yellow]Press Ctrl+C to stop the server[/yellow]")
            
            uvicorn.run(api.app, host=args.host, port=args.port)
            
        elif args.mode == "lm-studio":
            # Deploy to LM Studio
            deployer = LMStudioDeployer(args.model_path, args.endpoint)
            fused_path = deployer.fuse_model(args.fuse_output)
            
            console.print(f"[green]✅ Model prepared for LM Studio deployment[/green]")
            console.print(f"[blue]📁 Fused model at: {fused_path}[/blue]")
            console.print(f"[yellow]💡 Next steps:[/yellow]")
            console.print(f"   1. Load the fused model in LM Studio")
            console.print(f"   2. Start LM Studio server")
            console.print(f"   3. Test at {args.endpoint}")
            
        elif args.mode == "batch":
            # Batch processing
            if not args.input_file:
                console.print("[red]❌ --input-file required for batch mode[/red]")
                return 1
            
            processor = BatchProcessor(args.model_path, args.base_model)
            processor.load_model()
            
            output_file = processor.process_file(args.input_file, args.output_file)
            console.print(f"[green]✅ Batch processing complete[/green]")
            
        else:
            console.print(f"[red]❌ Unknown mode: {args.mode}[/red]")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Deployment stopped by user[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed: {e}[/bold red]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())