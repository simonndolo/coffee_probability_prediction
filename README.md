cat > README.md << 'EOF'
# Coffee Probability Prediction

## How to Run the Model

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the GUI application
python coffee_gui.py

### 3. Required files (must be in same directory)
- binary_best.pth - Model weights
- cultivation_best.pth - Model weights  
- model_pt.py - Model architecture
- coffee_model.py - Model utilities

**Note:** Model files (.pth) are not included in this repository and must be present locally.
EOF
