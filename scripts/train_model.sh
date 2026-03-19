#!/bin/bash
# LLM Trainer - Model Training Script
# Trains a specialized AI model using MLX LoRA fine-tuning

set -e

# Default configuration
MODEL="Qwen/Qwen2.5-3B-Instruct"
DATA_DIR="./data"
OUTPUT_DIR="./models/trained"
ITERATIONS=200
BATCH_SIZE=4
LEARNING_RATE="1e-5"
NUM_LAYERS=8
MODEL_NAME="business-ai-$(date +%Y%m%d-%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BOLD}$1${NC}"
}

# Help function
show_help() {
    cat << EOF
LLM Trainer - Model Training Script

Usage: $0 [OPTIONS]

Options:
    --model MODEL           Base model to fine-tune (default: $MODEL)
    --data-dir DIR          Directory containing training data (default: $DATA_DIR)
    --output-dir DIR        Directory to save trained model (default: $OUTPUT_DIR)
    --name NAME             Name for the trained model (default: auto-generated)
    --iterations N          Number of training iterations (default: $ITERATIONS)
    --batch-size N          Batch size for training (default: $BATCH_SIZE)
    --learning-rate RATE    Learning rate (default: $LEARNING_RATE)
    --num-layers N          Number of layers to apply LoRA (default: $NUM_LAYERS)
    --model-size SIZE       Model size (1B, 3B, 7B) - adjusts parameters automatically
    --quick                 Quick training for testing (50 iterations, smaller batch)
    --help                  Show this help message

Examples:
    $0                                    # Train with defaults
    $0 --name email-classifier           # Train with custom name
    $0 --quick                           # Quick training for testing
    $0 --model-size 1B                   # Use smaller model for limited RAM
    $0 --iterations 400 --batch-size 2   # Custom training parameters

EOF
}

# Check if virtual environment is activated
check_environment() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        print_warning "Virtual environment not detected"
        print_status "Attempting to activate local environment..."
        if [[ -f "./venv/bin/activate" ]]; then
            source ./venv/bin/activate
            print_success "Virtual environment activated"
        else
            print_error "No virtual environment found"
            print_status "Run ./scripts/setup.sh first"
            exit 1
        fi
    else
        print_success "Virtual environment active: $VIRTUAL_ENV"
    fi
}

# Check if training data exists
check_training_data() {
    local required_files=("train.jsonl" "valid.jsonl" "test.jsonl")
    
    print_status "Checking training data in $DATA_DIR..."
    
    for file in "${required_files[@]}"; do
        local filepath="$DATA_DIR/$file"
        if [[ ! -f "$filepath" ]]; then
            print_error "Missing training file: $filepath"
            print_status "Generate training data first:"
            print_status "  python scripts/generate_training_data.py --business-type consulting"
            exit 1
        else
            local count=$(wc -l < "$filepath")
            print_success "Found $filepath ($count examples)"
        fi
    done
}

# Check system resources
check_resources() {
    print_status "Checking system resources..."
    
    # Check available RAM (macOS)
    if command -v vm_stat >/dev/null 2>&1; then
        local free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
        local page_size=4096
        local free_ram_gb=$((free_pages * page_size / 1024 / 1024 / 1024))
        
        print_status "Available RAM: ~${free_ram_gb}GB"
        
        if [[ $free_ram_gb -lt 4 ]]; then
            print_warning "Low available RAM. Consider:"
            print_status "  - Closing other applications"
            print_status "  - Using --batch-size 1"
            print_status "  - Using --model-size 1B"
        fi
    fi
    
    # Check if this is Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        print_success "Apple Silicon detected - optimal for MLX"
    else
        print_warning "Non-Apple Silicon detected - training may be slower"
    fi
}

# Adjust parameters based on model size
adjust_for_model_size() {
    case $1 in
        "1B")
            MODEL="microsoft/DialoGPT-small"  # Example 1B model
            BATCH_SIZE=8
            NUM_LAYERS=4
            ;;
        "3B")
            MODEL="Qwen/Qwen2.5-3B-Instruct"
            BATCH_SIZE=4
            NUM_LAYERS=8
            ;;
        "7B")
            MODEL="microsoft/DialoGPT-large"  # Example 7B model
            BATCH_SIZE=2
            NUM_LAYERS=12
            ;;
        *)
            print_error "Unknown model size: $1"
            print_status "Valid sizes: 1B, 3B, 7B"
            exit 1
            ;;
    esac
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --name)
                MODEL_NAME="$2"
                shift 2
                ;;
            --iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --num-layers)
                NUM_LAYERS="$2"
                shift 2
                ;;
            --model-size)
                adjust_for_model_size "$2"
                shift 2
                ;;
            --quick)
                ITERATIONS=50
                BATCH_SIZE=2
                print_status "Quick training mode enabled"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Main training function
train_model() {
    print_header "🧠 Starting Model Training"
    print_status "Model: $MODEL"
    print_status "Output: $OUTPUT_DIR/$MODEL_NAME"
    print_status "Iterations: $ITERATIONS"
    print_status "Batch size: $BATCH_SIZE"
    print_status "Learning rate: $LEARNING_RATE"
    print_status "LoRA layers: $NUM_LAYERS"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR/$MODEL_NAME"
    
    # Create training log
    local log_file="$OUTPUT_DIR/$MODEL_NAME/training.log"
    local start_time=$(date)
    
    echo "Training started: $start_time" > "$log_file"
    echo "Model: $MODEL" >> "$log_file"
    echo "Parameters: iterations=$ITERATIONS, batch_size=$BATCH_SIZE, lr=$LEARNING_RATE, layers=$NUM_LAYERS" >> "$log_file"
    echo "" >> "$log_file"
    
    print_status "Starting training... (this may take 1-5 minutes)"
    print_status "Log file: $log_file"
    
    # Run MLX training
    python -m mlx_lm.lora \
        --model "$MODEL" \
        --train \
        --data "$DATA_DIR" \
        --iters "$ITERATIONS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --num-layers "$NUM_LAYERS" \
        --adapter-path "$OUTPUT_DIR/$MODEL_NAME/adapters" \
        2>&1 | tee -a "$log_file"
    
    local training_exit_code=${PIPESTATUS[0]}
    local end_time=$(date)
    
    echo "" >> "$log_file"
    echo "Training completed: $end_time" >> "$log_file"
    echo "Exit code: $training_exit_code" >> "$log_file"
    
    if [[ $training_exit_code -eq 0 ]]; then
        print_success "Training completed successfully!"
    else
        print_error "Training failed with exit code: $training_exit_code"
        print_status "Check log file for details: $log_file"
        exit 1
    fi
}

# Test the trained model
test_model() {
    print_header "🧪 Testing Trained Model"
    
    local adapter_path="$OUTPUT_DIR/$MODEL_NAME/adapters"
    
    if [[ ! -d "$adapter_path" ]]; then
        print_error "No adapters found at: $adapter_path"
        exit 1
    fi
    
    print_status "Testing with sample queries..."
    
    # Test queries
    local test_queries=(
        "Classify this business email: I'm interested in your consulting services"
        "Classify this business email: The system is down and we need help"
        "Classify this business email: Can we schedule a meeting next week?"
    )
    
    for query in "${test_queries[@]}"; do
        print_status "Query: $query"
        
        local response=$(python -m mlx_lm.generate \
            --model "$MODEL" \
            --adapter-path "$adapter_path" \
            --prompt "$query" \
            --max-tokens 100 \
            --temp 0.1 2>/dev/null | tail -n 1)
        
        print_success "Response: $response"
        echo ""
    done
}

# Create model info file
create_model_info() {
    local info_file="$OUTPUT_DIR/$MODEL_NAME/model_info.json"
    
    cat > "$info_file" << EOF
{
    "name": "$MODEL_NAME",
    "base_model": "$MODEL",
    "training_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "training_parameters": {
        "iterations": $ITERATIONS,
        "batch_size": $BATCH_SIZE,
        "learning_rate": "$LEARNING_RATE",
        "num_layers": $NUM_LAYERS
    },
    "data_info": {
        "data_directory": "$DATA_DIR",
        "train_examples": $(wc -l < "$DATA_DIR/train.jsonl"),
        "valid_examples": $(wc -l < "$DATA_DIR/valid.jsonl"),
        "test_examples": $(wc -l < "$DATA_DIR/test.jsonl")
    },
    "system_info": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "hostname": "$(hostname)"
    }
}
EOF

    print_success "Model info saved: $info_file"
}

# Main execution
main() {
    print_header "🚀 LLM Trainer - Model Training"
    
    # Parse arguments
    parse_args "$@"
    
    # Preflight checks
    check_environment
    check_training_data
    check_resources
    
    # Training process
    train_model
    test_model
    create_model_info
    
    print_header "🎉 Training Complete!"
    print_success "Model saved to: $OUTPUT_DIR/$MODEL_NAME"
    print_status "Next steps:"
    print_status "  1. Test thoroughly: python scripts/test_model.py --model-path $OUTPUT_DIR/$MODEL_NAME"
    print_status "  2. Deploy for production: python scripts/deploy_model.py --model-path $OUTPUT_DIR/$MODEL_NAME"
    print_status "  3. Monitor performance: python scripts/monitor_performance.py --model-path $OUTPUT_DIR/$MODEL_NAME"
}

# Run main function
main "$@"