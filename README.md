## SubSec
Sub-second LLM inference through aggressive and adaptive optimization.

### Granite-4 Family Support

SubSec is tuned to run the IBM Granite-4 family on low latency GPUs (e.g., NVIDIA A10), with
special handling for both dense attention-only models and hybrid Mamba + Attention models:

- **Dense (attention-only) Granite-4 models**
  - `ibm-granite/granite-4.0-micro-base`
  - `ibm-granite/granite-4.0-micro`

- **Hybrid (Mamba + Attention) Granite-4-H models**
  - `ibm-granite/granite-4.0-h-micro-base`
  - `ibm-granite/granite-4.0-h-micro`
  - `ibm-granite/granite-4.0-h-tiny-base`
  - `ibm-granite/granite-4.0-h-tiny`
  - `ibm-granite/granite-4.0-h-small-base`
  - `ibm-granite/granite-4.0-h-small`

Hybrid models (those with `-h-` in the name) use Mamba-2 style state space layers alongside
Transformer attention. SubSec enables:

- **Attention kernels**
  - FlashAttention 2 where available (`attn_implementation="flash_attention_2"`).
  - Fallback to standard `eager` attention when FlashAttention 2 is not installed.

- **Mamba / SSM kernels (for Granite-4-H)**
  - Uses the fused CUDA kernels from `mamba-ssm` and `causal-conv1d` when they are available.
  - Falls back to slower pure-PyTorch implementations if those libraries are missing.

At startup, `inference.py` prints a short **SubSec Model Summary** that shows:

- **Model ID** (e.g., `ibm-granite/granite-4.0-h-micro`)
- **Architecture**, context window, hidden size, and layer count (from the model config)
- Whether it is **Granite-4** and whether it is **Hybrid (Mamba + Attention)**
- Whether **FlashAttention 2** and **Mamba-SSM fused kernels** are active

This summary plus the per-query latency metrics gives you everything you need to demonstrate
that “Granite-4 is fully supported on SubSec” with a concrete streaming and latency demo.

### Environment Setup

Everything is expected to run inside a Conda environment called `subsec`.
Use the helper script to create and configure it:

```bash
cd SubSec
source subsec_setup.sh
```

The script:

- **Installs Anaconda** (if needed) and pins a recent `conda` and Python.
- **Creates/updates** the `subsec` environment.
- Installs:
  - **PyTorch** with CUDA support where possible (falling back to CPU-only if needed).
  - **transformers** and **accelerate** for model loading and inference.
  - **mamba-ssm**, **causal-conv1d**, and **einops** for hybrid Granite-4-H models
    (optional but strongly recommended).
  - **flash-attn** build dependencies and the library itself (best-effort, optional).

If Mamba-SSM or FlashAttention 2 are not available for your platform, SubSec will keep working,
but latency will be higher for attention and/or Mamba layers.

### Running Inference

Activate your environment:

```bash
conda activate subsec
cd SubSec
```

Then run the optimized SubSec chat interface:

```bash
# Default model (Granite-4.0 micro, dense transformer)
python inference.py

# Or explicitly choose any Granite-4 model via environment variable
MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-h-micro python inference.py
MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-h-small python inference.py
MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-micro-base python inference.py
```

For hybrid Granite-4-H models, `inference.py` automatically:

- Enables **remote model code** so the custom Mamba + Attention architecture is used.
- Detects whether **Mamba-SSM** and **causal-conv1d** fused kernels are available.

You can override the remote code behavior (for any model) via:

```bash
SUBSEC_TRUST_REMOTE_CODE=1 MODEL_NAME_OR_PATH=some/model python inference.py
```

### Streaming and Latency Metrics

The SubSec chat interface streams tokens in real time and prints latency metrics
for each turn:

- **TTFT (Time To First Token)** in milliseconds.
- **Total tokens generated**.
- **Total time** and **tokens/sec** throughput.

This makes it easy to benchmark different Granite-4 models (micro vs tiny vs small,
hybrid vs dense) on your NVIDIA A10 GPU and to capture screenshots or logs for
sharing performance results (for example in a LinkedIn post or technical write-up).
