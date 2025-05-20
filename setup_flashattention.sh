# Ensure the repository is cloned only if it doesn't exist, and change into the repository folder
mkdir -p build
cd build

# Check if the 'flash-attention' repository already exists
if [ ! -d "flash-attention" ]; then
    echo "Cloning flash-attention repository..."
    git clone https://github.com/Dao-AILab/flash-attention.git
else
    echo "flash-attention repository already exists. Skipping clone."
fi

# Navigate to the 'flash-attention' directory
cd flash-attention 

# Navigate to the 'hopper' subdirectory
cd hopper/

# Install the package
export MAX_JOBS=4
python setup.py install

echo "Verify Flash Attention Install!"

# --- verify flash-attn installation ---
python - << 'EOF'
import sys
try:
    import flash_attn_interface                          
    print("flash_attn_func:", flash_attn_interface.flash_attn_func)
except Exception as e:
    print("flash_attn import failed:", e, file=sys.stderr)
    sys.exit(1)
EOF
# --------------------------------------
