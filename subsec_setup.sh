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

    # Install PyTorch and torchvision via conda first (to get compatible MKL)
    print_info "Installing PyTorch and torchvision via conda (pre-built binaries, faster)..."
    conda install -y -c pytorch pytorch torchvision

    # Check if we have MKL compatibility issues and fix if needed
    print_info "Checking MKL compatibility..."
    if ! python -c "import torch; print('PyTorch import successful')" 2>/dev/null; then
        print_warning "PyTorch import failed, trying alternative MKL configuration..."
        # Try using OpenBLAS instead of MKL to avoid compatibility issues
        conda install -y -c conda-forge openblas blas=2.116
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    fi

    # Install build dependencies for compiling flash-attn
    print_info "Installing build dependencies (compilers, build tools)..."
    conda install -y -c conda-forge ninja cmake gcc_linux-64 gxx_linux-64

    # Set parallel compilation flags for faster builds
    NUM_CORES=$(nproc 2>/dev/null || echo "4")
    export MAX_JOBS="$NUM_CORES"
    export CMAKE_BUILD_PARALLEL_LEVEL="$NUM_CORES"

    # Install transformers and accelerate (needed for inference.py)
    print_info "Installing transformers and accelerate..."
    pip install transformers accelerate

    # Install additional packages needed for flash-attn build
    print_info "Installing packaging and wheel for flash-attn build..."
    pip install packaging wheel
    
    # Install flash-attn from source to ensure ABI compatibility with conda PyTorch
    # Building from source ensures binary compatibility with the installed PyTorch version
    print_info "Installing flash-attn from source (this may take 5-10 minutes)..."
    print_info "This builds flash-attn against your PyTorch to ensure ABI compatibility..."
    pip install 'git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3#egg=flash-attn&subdirectory=.' --no-build-isolation || {
        print_warning "Flash-attn installation failed. This may impact performance but won't prevent the model from running."
        print_warning "You can try installing it later with: pip install 'git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3#egg=flash-attn&subdirectory=.' --no-build-isolation"
    }

    # Install Mamba SSM kernels for hybrid Mamba-Attention models (like Granite)
    # These provide optimized CUDA kernels for Mamba state space model layers
    print_info "Installing causal-conv1d from source (this may take 2-3 minutes)..."
    pip install causal-conv1d --no-build-isolation || {
        print_warning "causal-conv1d installation failed. Mamba layers will use slower fallback implementation."
    }

    print_info "Installing mamba-ssm from source (this may take 3-5 minutes)..."
    pip install mamba-ssm --no-build-isolation || {
        print_warning "mamba-ssm installation failed. Mamba layers will use slower fallback implementation."
    }

    print_success "All dependencies installed successfully!"
}

# Main function
main() {
    # Exit on error
    set -e
    
    setup_conda
    setup_environment
    
    print_step "✅ Setup Complete!"
    print_info "To activate the environment in future sessions, run: conda activate subsec"
}

# Run main function
main "$@"
