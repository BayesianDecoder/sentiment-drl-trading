#!/bin/bash
# ══════════════════════════════════════════════════════
# AuraADAS / Sentiment DRL Trading — Environment Setup
# Run this ONCE on your machine or Kaggle/Colab
# ══════════════════════════════════════════════════════



# Step 1: Create virtual environment (skip on Colab/Kaggle)
if [ -z "$KAGGLE_KERNEL_RUN_TYPE" ] && [ -z "$COLAB_GPU" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Step 2: Upgrade pip
pip install --upgrade pip --quiet

# Step 3: Install requirements

pip install -r requirements.txt --quiet


# Step 4: Verify key imports

python3 -c "
import torch
import transformers
import peft
import sklearn
import mlflow
import datasets

print(f'  PyTorch     : {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT        : {peft.__version__}')
print(f'  Sklearn     : {sklearn.__version__}')
print(f'  MLflow      : {mlflow.__version__}')
print(f'  Datasets    : {datasets.__version__}')
print(f'  GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU name     : {torch.cuda.get_device_name(0)}')

"

# Step 5: Create MLflow experiment
echo ""
echo "Setting up MLflow experiment..."
python3 -c "
import mlflow
mlflow.set_experiment('IndianFinSentiment')

"


