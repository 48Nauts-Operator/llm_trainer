#!/usr/bin/env python3
"""
LLM Trainer - Training Data Generator
Generates business-specific training data for fine-tuning AI models.
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

# Import business templates
def load_business_templates():
    """Load all available business templates"""
    templates = {}
    templates_dir = Path(__file__).parent.parent / "templates"
    
    # Default templates if files don't exist
    templates["consulting"] = {
        "sales_inquiry": {
            "patterns": [
                "I'm interested in your {service} services",
                "Can you help us with {service}?", 
                "What are your rates for {service}?",
                "We need help with {service}",
                "I'd like to discuss {service} options",
                "Can you provide {service} consulting?",
                "We're looking for {service} expertise",
                "Do you offer {service} solutions?"
            ],
            "services": [
                "strategic planning", "digital transformation", "process optimization",
                "change management", "organizational design", "performance improvement",
                "business analysis", "project management", "leadership development"
            ],
            "response": {"priority": "high", "action": "personal_response", "department": "sales"}
        },
        "support_request": {
            "patterns": [
                "I'm having trouble with {issue}",
                "The {system} isn't working as expected",
                "Can you help troubleshoot {issue}?",
                "We need technical support for {system}",
                "There's a problem with {issue}",
                "The {system} has stopped working",
                "Help needed with {issue}"
            ],
            "systems": [
                "implementation", "system", "process", "workflow", "solution",
                "platform", "integration", "configuration", "deployment"
            ],
            "response": {"priority": "medium", "action": "assign_support", "department": "support"}
        },
        "meeting_request": {
            "patterns": [
                "Can we schedule a call to discuss {topic}?",
                "Would you be available for a meeting about {topic}?",
                "I'd like to set up a consultation on {topic}",
                "Let's arrange a discovery call for {topic}",
                "When can we meet to discuss {topic}?",
                "I need a meeting about {topic}",
                "Can we book time to talk about {topic}?"
            ],
            "topics": [
                "our project", "next steps", "implementation", "strategy",
                "requirements", "timeline", "budget", "proposal", "partnership"
            ],
            "response": {"priority": "high", "action": "schedule_meeting", "department": "sales"}
        }
    }
    
    templates["ecommerce"] = {
        "vip_customer": {
            "patterns": [
                "Customer has spent ${amount}+ this year",
                "Customer orders {product_type} regularly",
                "Customer has been with us for {duration}",
                "Premium customer with {history}",
                "Long-term customer showing {behavior}",
                "High-value customer with {pattern}"
            ],
            "amounts": ["5000", "10000", "15000", "20000"],
            "product_types": ["premium products", "enterprise solutions", "bulk orders"],
            "durations": ["3+ years", "5+ years", "2+ years"],
            "histories": ["excellent payment history", "multiple large orders"],
            "behaviors": ["consistent purchasing", "strong loyalty"],
            "patterns": ["increasing order sizes", "regular reorders"],
            "response": {"segment": "vip", "discount": "20%", "support": "priority"}
        },
        "price_sensitive": {
            "patterns": [
                "Customer only buys during {event}",
                "Customer always uses {discount_type}",
                "Customer {behavior} extensively",
                "Budget-conscious customer who {action}",
                "Price-focused buyer with {pattern}"
            ],
            "events": ["sales", "promotions", "clearance events"],
            "discount_types": ["discount codes", "coupons", "promotional offers"],
            "behaviors": ["compares prices", "waits for sales"],
            "actions": ["shops clearance", "uses every coupon"],
            "patterns": ["minimal spend", "deal-hunting behavior"],
            "response": {"segment": "price_sensitive", "discount": "15%", "support": "standard"}
        }
    }
    
    templates["medical"] = {
        "urgent_appointment": {
            "patterns": [
                "I'm experiencing {symptom}",
                "My {who} has {symptom}",
                "Emergency: {symptom}",
                "I think I {condition}",
                "Urgent: {symptom}",
                "Please help: {symptom}"
            ],
            "symptoms": [
                "severe chest pain", "difficulty breathing", "high fever",
                "severe bleeding", "loss of consciousness", "severe allergic reaction"
            ],
            "who": ["child", "parent", "spouse"],
            "conditions": ["broke my arm", "had a stroke", "am having an allergic reaction"],
            "response": {"priority": "emergency", "timeframe": "today", "department": "urgent_care"}
        },
        "routine_checkup": {
            "patterns": [
                "I need my {appointment_type}",
                "Time for my {routine}",
                "I need a {exam_type}",
                "Can I schedule my {checkup}?"
            ],
            "appointment_types": ["annual physical", "routine checkup", "wellness visit"],
            "routines": ["routine blood work", "annual screening", "preventive care"],
            "exam_types": ["wellness exam", "health screening", "preventive checkup"],
            "checkups": ["yearly exam", "routine appointment", "health maintenance"],
            "response": {"priority": "routine", "timeframe": "within_month", "department": "primary_care"}
        }
    }
    
    return templates

def generate_variations(pattern: str, replacements: Dict[str, List[str]], count: int = 5) -> List[str]:
    """Generate variations of a pattern with random replacements"""
    variations = []
    for _ in range(count):
        varied_pattern = pattern
        for placeholder, options in replacements.items():
            if f"{{{placeholder}}}" in varied_pattern:
                varied_pattern = varied_pattern.replace(f"{{{placeholder}}}", random.choice(options))
        variations.append(varied_pattern)
    return variations

def generate_business_data(business_type: str, examples_per_category: int = 100) -> List[Dict[str, str]]:
    """Generate training data for a specific business type"""
    templates = load_business_templates()
    
    if business_type not in templates:
        raise ValueError(f"Business type '{business_type}' not found. Available: {list(templates.keys())}")
    
    business_template = templates[business_type]
    training_data = []
    
    console.print(f"[blue]Generating {examples_per_category} examples per category for {business_type}...[/blue]")
    
    for category, category_data in business_template.items():
        patterns = category_data["patterns"]
        response = category_data["response"]
        
        # Extract replacement options
        replacements = {k: v for k, v in category_data.items() 
                       if k not in ["patterns", "response"]}
        
        for _ in track(range(examples_per_category), description=f"Category: {category}"):
            # Choose a random pattern
            pattern = random.choice(patterns)
            
            # Generate variations if there are placeholders
            if "{" in pattern and replacements:
                variations = generate_variations(pattern, replacements, 1)
                final_text = variations[0]
            else:
                final_text = pattern
            
            # Add context variations
            contexts = [
                f"Classify this business email: {final_text}",
                f"Categorize this inquiry: {final_text}",
                f"Route this request: {final_text}",
                f"Process this message: {final_text}",
            ]
            
            training_data.append({
                "prompt": random.choice(contexts),
                "completion": json.dumps(response)
            })
    
    return training_data

def split_data(data: List[Dict[str, str]], train_ratio: float = 0.8, 
               valid_ratio: float = 0.1) -> tuple:
    """Split data into train, validation, and test sets"""
    random.shuffle(data)
    total = len(data)
    
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    
    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    
    return train_data, valid_data, test_data

def save_data(data: List[Dict[str, str]], filepath: Path):
    """Save data to JSONL format"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def create_custom_template():
    """Interactive creation of custom business template"""
    console.print("[yellow]Creating custom business template...[/yellow]")
    
    business_name = console.input("Business name/type: ")
    categories = {}
    
    while True:
        category_name = console.input(f"\nCategory name (or 'done' to finish): ")
        if category_name.lower() == 'done':
            break
            
        console.print(f"\nDefining category: [bold]{category_name}[/bold]")
        
        patterns = []
        console.print("Enter example patterns (use {placeholder} for variables):")
        while True:
            pattern = console.input(f"Pattern {len(patterns) + 1} (or empty to finish): ")
            if not pattern.strip():
                break
            patterns.append(pattern)
        
        # Get response structure
        console.print("\nDefine the classification response:")
        response = {}
        while True:
            key = console.input("Response key (or empty to finish): ")
            if not key.strip():
                break
            value = console.input(f"Value for '{key}': ")
            response[key] = value
        
        categories[category_name] = {
            "patterns": patterns,
            "response": response
        }
    
    # Save custom template
    template_file = Path(__file__).parent.parent / "templates" / f"{business_name.lower()}.py"
    template_file.parent.mkdir(exist_ok=True)
    
    with open(template_file, 'w') as f:
        f.write(f'# Custom template for {business_name}\n')
        f.write(f'TEMPLATE = {json.dumps(categories, indent=2)}\n')
    
    console.print(f"[green]Custom template saved to {template_file}[/green]")
    return categories

def main():
    parser = argparse.ArgumentParser(description="Generate training data for business AI models")
    parser.add_argument("--business-type", choices=["consulting", "ecommerce", "medical", "legal", "custom"],
                       default="consulting", help="Type of business to generate data for")
    parser.add_argument("--examples-per-category", type=int, default=100,
                       help="Number of examples to generate per category")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Directory to save training data")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Proportion of data for training")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                       help="Proportion of data for validation")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    console.print(f"[bold green]LLM Trainer - Data Generator[/bold green]")
    console.print(f"Business type: {args.business_type}")
    console.print(f"Examples per category: {args.examples_per_category}")
    
    try:
        # Generate data
        if args.business_type == "custom":
            business_data = create_custom_template()
            # Would need to modify generate_business_data to handle custom templates
            console.print("[yellow]Custom template created. Run again with the saved template.[/yellow]")
            return
        else:
            data = generate_business_data(args.business_type, args.examples_per_category)
        
        # Split data
        train_data, valid_data, test_data = split_data(
            data, args.train_ratio, args.valid_ratio
        )
        
        # Save data
        output_dir = Path(args.output_dir)
        save_data(train_data, output_dir / "train.jsonl")
        save_data(valid_data, output_dir / "valid.jsonl")
        save_data(test_data, output_dir / "test.jsonl")
        
        # Display summary
        table = Table(title="Training Data Summary")
        table.add_column("Split", style="cyan")
        table.add_column("Examples", style="magenta")
        table.add_column("File", style="green")
        
        table.add_row("Train", str(len(train_data)), str(output_dir / "train.jsonl"))
        table.add_row("Validation", str(len(valid_data)), str(output_dir / "valid.jsonl"))
        table.add_row("Test", str(len(test_data)), str(output_dir / "test.jsonl"))
        table.add_row("Total", str(len(data)), "")
        
        console.print(table)
        console.print(f"\n[bold green]✅ Data generation complete![/bold green]")
        console.print(f"Next step: ./scripts/train_model.sh")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())