#!/bin/bash

source ~/.bashrc

# Helper functions for colored output
print_step() {
    echo -e "\n\033[1;34m$1\033[0m"
}

print_info() {
    echo -e "\033[1;36mℹ $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✓ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠ $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m✗ $1\033[0m" >&2
}

# Function to check if Anaconda (not Miniconda) is installed and install if needed
setup_conda() {
    print_step "🐍 STEP 1: Setting up Conda Environment"

    local INSTALLER="Anaconda3-2024.02-1-Linux-x86_64.sh"
    local URL="https://repo.anaconda.com/archive/${INSTALLER}"
    local PREFIX="$HOME/anaconda3"

    # Arch check
    case "$(uname -m)" in
        x86_64|amd64) : ;;
        *) print_error "This Anaconda installer is for x86_64/AMD64; detected '$(uname -m)'."; exit 1;;
    esac

    # If Anaconda already present at expected path with correct versions, use it
    if [[ -x "$PREFIX/bin/conda" ]]; then
        local cur_conda="$("$PREFIX/bin/conda" --version | awk '{print $2}')"
        local cur_py="$("$PREFIX/bin/python" -V 2>&1 | awk '{print $2}')"
        if [[ "$cur_conda" == "24.5.0" && "$cur_py" == "3.11.7" ]]; then
            print_success "Anaconda already installed at $PREFIX (Python $cur_py, conda $cur_conda)"
            eval "$("$PREFIX/bin/conda" shell.bash hook)"
            return 0
        fi
        print_info "Anaconda found at $PREFIX but versions differ (Python $cur_py, conda $cur_conda). Updating…"
    else
        # If some other conda exists (e.g., Miniconda), warn but proceed with Anaconda install
        if command -v conda >/dev/null 2>&1; then
            print_warning "Another Conda installation detected: $(conda --version). Installing Anaconda to $PREFIX anyway."
        fi

        print_info "Downloading Anaconda Distribution 2024.02-1 (Python 3.11.7) for Linux x86_64…"
        local TMP="/tmp/$INSTALLER"
        if command -v curl >/dev/null 2>&1; then
            curl -fL "$URL" -o "$TMP"
        else
            wget -q "$URL" -O "$TMP"
        fi

        print_info "Installing to $PREFIX (batch mode)…"
        bash "$TMP" -b -p "$PREFIX"
        rm -f "$TMP"
    fi

    # Initialize conda for this shell and future shells
    eval "$("$PREFIX/bin/conda" shell.bash hook)"
    "$PREFIX/bin/conda" init bash >/dev/null 2>&1 || true

    # Pin conda to the requested version
    print_info "Pinning conda to 24.5.0 in base…"
    conda install -n base -y "conda=24.5.0"

    # Ensure PATH for future sessions
    if ! grep -q 'anaconda3/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> "$HOME/.bashrc"
    fi

    # Optional: avoid auto-activating base in future shells
    conda config --set auto_activate_base false >/dev/null 2>&1 || true

    # Sanity logs
    local pyv; pyv="$(python -V 2>&1 | awk '{print $2}')"
    local cv;  cv="$(conda --version | awk '{print $2}')"
    print_success "Anaconda ready (Python $pyv, conda $cv)."
}

# Function to create and set up the conda environment
setup_environment() {
    print_step "📦 STEP 2: Creating Conda Environment 'subsec'"

    # Check if environment already exists
    if conda env list | grep -q "^subsec "; then
        print_info "Environment 'subsec' already exists. Skipping creation."
    else
        print_info "Creating conda environment 'subsec' with Python 3.11..."
        conda create -n subsec python=3.11 -y
        print_success "Environment 'subsec' created successfully."
    fi

    # Activate the environment
    print_step "🔧 STEP 3: Activating Environment and Installing Dependencies"
    print_info "Activating conda environment 'subsec'..."
    eval "$(conda shell.bash hook)"
    conda activate subsec
    
    # Verify we're using the right Python
    if ! python -c "import sys; assert 'subsec' in sys.executable or 'subsec' in sys.prefix" 2>/dev/null; then
        print_warning "Warning: May not be using the 'subsec' environment Python"
    fi
    print_info "Using Python: $(python --version) at $(which python)"

    # Install PyTorch with CUDA support - use compatible versions from PyPI
    # We don't install torchvision since we don't need it for text-only models
    # This prevents the torchvision::nms operator error
    print_info "Installing PyTorch with CUDA support (without torchvision)..."
    if ! pip install torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null; then
        print_warning "CUDA 12.1 PyTorch installation failed, trying CUDA 11.8..."
        if ! pip install torch --index-url https://download.pytorch.org/whl/cu118 2>/dev/null; then
            print_warning "CUDA PyTorch installation failed, installing CPU-only version..."
            pip install torch
        fi
    fi

    # Verify PyTorch installation
    if ! python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" 2>/dev/null; then
        print_error "PyTorch installation verification failed!"
        return 1
    fi

    # Install core dependencies for streaming inference
    print_info "Installing transformers and accelerate..."
    pip install transformers accelerate

    # Install Mamba-2 / SSM dependencies for hybrid Granite-4 models
    # These provide fused GPU kernels for the Mamba layers used in hybrid Granite-4.x-H models.
    # NOTE: This is the Mamba SSM library (mamba-ssm), NOT the conda solver called 'mamba'.
    print_info "Installing Mamba-SSM fused kernels and causal-conv1d for hybrid Granite-4 models (optional but recommended)..."
    if ! pip install "mamba-ssm>=2.0.0" "causal-conv1d>=1.0.0" einops 2>/dev/null; then
        print_warning "Mamba-SSM / causal-conv1d installation failed or is unavailable for this platform."
        print_warning "Granite-4 hybrid (H) models will still run but Mamba layers may be slower (fallback PyTorch kernels)."
    fi

    # Verify that Mamba-SSM and causal-conv1d are importable (for hybrid models)
    python - << 'PY' 2>/dev/null
try:
    import mamba_ssm  # type: ignore
    import causal_conv1d  # type: ignore
    print("Mamba-SSM and causal-conv1d detected for hybrid Granite-4 models.")
except Exception as e:
    print("Warning: Optimized Mamba-SSM / causal-conv1d kernels not fully available:", e)
PY

    # Install build dependencies for flash-attn (optional, for performance)
    print_info "Installing build dependencies for flash-attn (optional)..."
    conda install -y -c conda-forge ninja cmake gcc_linux-64 gxx_linux-64 2>/dev/null || {
        print_warning "Could not install build dependencies via conda, trying pip..."
        pip install ninja cmake 2>/dev/null || print_warning "Build dependencies not available, flash-attn may not compile"
    }

    # Set parallel compilation flags for faster builds
    NUM_CORES=$(nproc 2>/dev/null || echo "4")
    export MAX_JOBS="$NUM_CORES"
    export CMAKE_BUILD_PARALLEL_LEVEL="$NUM_CORES"

    # Try to install flash-attn (optional, for better performance)
    # This is optional - the code will work without it
    print_info "Attempting to install flash-attn (optional, may take 5-10 minutes)..."
    pip install 'git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3#egg=flash-attn&subdirectory=.' --no-build-isolation 2>/dev/null || {
        print_warning "Flash-attn installation skipped (optional - code will work without it)"
    }

    print_success "All dependencies installed successfully!"
}

# Main function
main() {
    # Don't exit on error - we want to handle errors gracefully
    set +e
    
    setup_conda
    if [ $? -ne 0 ]; then
        print_error "Conda setup failed!"
        exit 1
    fi
    
    setup_environment
    if [ $? -ne 0 ]; then
        print_error "Environment setup failed!"
        exit 1
    fi
    
    print_step "✅ Setup Complete!"
    print_info "To activate the environment in future sessions, run: conda activate subsec"
    print_info "You can now run: python inference_baseline.py or python inference.py"
}

# Run main function
main "$@"
