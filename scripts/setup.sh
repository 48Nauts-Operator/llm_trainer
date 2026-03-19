#!/bin/bash
# LLM Trainer - Environment Setup Script
# Sets up Python environment and installs dependencies

set -e  # Exit on any error

echo "🚀 Setting up LLM Trainer environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This toolkit is optimized for macOS with Apple Silicon"
    echo "   It may work on other platforms but performance will vary"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
required_version="3.10"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "✅ Python $python_version found"
else
    echo "❌ Python 3.10+ required. Found: $python_version"
    echo "   Install Python 3.10+ and try again"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install MLX (Apple Silicon optimized)
echo "🧠 Installing MLX framework..."
if [[ $(uname -m) == "arm64" ]]; then
    pip install mlx mlx-lm
    echo "✅ MLX installed for Apple Silicon"
else
    echo "⚠️  Installing MLX for non-Apple Silicon (may be slower)"
    pip install mlx mlx-lm
fi

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    asyncio \
    httpx \
    fastapi \
    uvicorn \
    pydantic \
    click \
    tqdm \
    rich

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import mlx
import mlx.core as mx
import mlx.nn as nn
print('✅ MLX imported successfully')

try:
    from mlx_lm import load, generate
    print('✅ MLX-LM imported successfully')
except ImportError as e:
    print(f'❌ MLX-LM import failed: {e}')
    exit(1)

print('✅ All dependencies verified')
"

# Test basic MLX functionality
echo "🧪 Testing MLX functionality..."
python3 -c "
import mlx.core as mx
import numpy as np

# Test basic tensor operations
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = a + b
print(f'✅ MLX tensor operations working: {c}')

# Test GPU/unified memory
try:
    # This will use unified memory on Apple Silicon
    large_array = mx.random.normal((1000, 1000))
    result = mx.sum(large_array)
    print(f'✅ MLX unified memory working: sum={result:.2f}')
except Exception as e:
    print(f'⚠️  Unified memory test warning: {e}')
"

# Create example config file
echo "⚙️  Creating default configuration..."
cat > config.yaml << 'EOF'
# LLM Trainer Configuration

# Default model settings
default_model: "Qwen/Qwen2.5-3B-Instruct"
model_cache_dir: "./models"

# Training settings
training:
  default_iterations: 200
  default_batch_size: 4
  default_learning_rate: 1e-5
  default_num_layers: 8

# Data settings
data:
  train_split: 0.8
  valid_split: 0.1
  test_split: 0.1
  min_examples_per_category: 50

# Performance settings
performance:
  temperature: 0.1  # Low for consistent classification
  max_tokens: 100   # Sufficient for JSON responses
  timeout_seconds: 10

# Monitoring settings
monitoring:
  log_level: "INFO"
  metrics_file: "./metrics.jsonl"
  accuracy_threshold: 0.90
EOF

# Create directories for data and models
echo "📁 Creating directory structure..."
mkdir -p data/{train,valid,test,raw}
mkdir -p models/{base,trained,checkpoints}
mkdir -p logs
mkdir -p output

# Create activation script
echo "📝 Creating activation script..."
cat > activate.sh << 'EOF'
#!/bin/bash
# Activate LLM Trainer environment
source venv/bin/activate
echo "✅ LLM Trainer environment activated"
echo "💡 Run './scripts/train_model.sh --help' to get started"
EOF
chmod +x activate.sh

# Create deactivation info
cat > deactivate_info.txt << 'EOF'
To deactivate the environment later, run:
    deactivate

To reactivate, run:
    source venv/bin/activate
    # or
    ./activate.sh
EOF

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment:    ./activate.sh"
echo "2. Generate training data:  python scripts/generate_training_data.py --business-type consulting"
echo "3. Train your model:        ./scripts/train_model.sh"
echo "4. Test the results:        python scripts/test_model.py"
echo ""
echo "📖 For detailed instructions, see docs/complete-guide.md"
echo ""