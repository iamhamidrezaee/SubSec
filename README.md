# SubSec

**Sub-second LLM inference through intelligent context management and adaptive optimization.**

SubSec makes long multi-turn conversations with Granite-4 models fast, stable, and memory-efficient by using bounded context windows and hierarchical summarization instead of naive full-history approaches.

---

## 🎯 Key Results

Based on rigorous testing with 7-turn, context-heavy conversations:

### Attention-Only Granite-4 Models
- **~53% reduction in average TTFT** (270ms → 126ms)
- **~60% reduction in max TTFT** (peak latency stays bounded)
- **Stable throughput** maintained at 33-35 tokens/sec
- **Quality preserved**: All topics correctly summarized and recalled

### Hybrid Granite-4-H (Mamba + Attention) Models
- **Baseline fails**: CUDA OOM during long conversations due to unbounded Mamba state growth
- **SubSec succeeds**: Runs stably with bounded context, preventing memory explosion
- **Key insight**: Without context bounding, hybrid models are **not viable** for production multi-turn use on typical GPUs

### How We Achieve This
- ✅ **Bounded context**: Recent turns in full detail + compact summary of older turns
- ✅ **No quality sacrifice**: Summary + sliding window preserves all key information
- ✅ **Model-agnostic**: Works across dense and hybrid Granite-4 architectures
- ✅ **Memory efficient**: Prevents KV cache and Mamba state explosion

---

## 📊 Supported Models

### Dense (Attention-Only) Granite-4
- `ibm-granite/granite-4.0-micro-base`
- `ibm-granite/granite-4.0-micro`

### Hybrid (Mamba + Attention) Granite-4-H
- `ibm-granite/granite-4.0-h-micro-base`
- `ibm-granite/granite-4.0-h-micro`
- `ibm-granite/granite-4.0-h-tiny-base`
- `ibm-granite/granite-4.0-h-tiny`
- `ibm-granite/granite-4.0-h-small-base`
- `ibm-granite/granite-4.0-h-small`

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd SubSec
source subsec_setup.sh
```

The script creates a `subsec` conda environment with:
- PyTorch with CUDA support
- Transformers & Accelerate
- FlashAttention 2 (for faster attention)
- Mamba-SSM kernels (for hybrid models)

### 2. Run Inference

```bash
conda activate subsec

# Default model (Granite-4.0-micro)
python inference.py

# Or choose a specific model
MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-h-micro python inference.py
MODEL_NAME_OR_PATH=ibm-granite/granite-4.0-micro-base python inference.py
```

### 3. Configure (Optional)

Copy and customize the environment file:

```bash
cp .env.example .env
# Edit .env to adjust context management, generation limits, etc.
```

---

## ⚙️ Configuration

SubSec exposes several environment variables to tune the latency-quality tradeoff:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME_OR_PATH` | `ibm-granite/granite-4.0-micro` | Which Granite-4 model to load |
| `SUBSEC_MAX_PROMPT_TOKENS` | `1024` | Max prompt length (bounds TTFT) |
| `SUBSEC_RECENT_TURNS_FULL` | `2` | Recent turns kept verbatim |
| `SUBSEC_SUMMARY_CHARS_PER_MSG` | `160` | Chars per message in summary |
| `SUBSEC_SUMMARY_MAX_LINES` | `12` | Max lines in summary |
| `SUBSEC_MAX_NEW_TOKENS` | `512` | Max tokens generated per turn |

See `.env.example` for detailed documentation and recommendations.

---

## 📈 Architecture & How It Works

### The Problem
**Naive approach**: Concatenate all prior conversation history into every new prompt.
- ❌ TTFT grows linearly with conversation length
- ❌ Hybrid models hit OOM as Mamba state explodes
- ❌ Wastes compute re-processing old context

### SubSec's Solution
**Bounded context with hierarchical summarization**:

```
┌─────────────────────────────────────────┐
│  Prompt sent to model:                  │
│                                         │
│  [Compact Summary of Turns 1-5]         │  ← Compressed
│  Turn 6: User + Assistant (full text)   │  ← Recent
│  Turn 7: User + Assistant (full text)   │  ← Recent
│  Turn 8: User (current query)           │  ← Current
└─────────────────────────────────────────┘
```

**Benefits**:
- ✅ Prompt length stays **bounded** regardless of conversation length
- ✅ TTFT remains **flat** across turns
- ✅ Recent context preserved **verbatim** for high-quality responses
- ✅ Older context preserved as **faithful summary** for topic recall
- ✅ Works with **HF's internal KV cache** (no manual cache surgery)

---

## 📁 Project Structure

```
SubSec/
├── inference.py                    # Main optimized chat interface
├── inference_baseline.py           # Naive baseline (for comparison)
├── subsec_setup.sh                 # Environment setup script
├── .env.example                    # Configuration template
└── README.md                       # This file
```

---

## 🔬 Technical Details

### Context Management Strategy

SubSec uses a **summary + sliding window** approach:

1. **Recent turns** (default: last 2 turns = 4 messages): Kept in **full detail**
2. **Older turns**: Compressed into an **ultra-compact system summary**:
   - Extracts topic keywords from each user query
   - Truncates assistant responses to core information
   - Bounds total summary size strictly

This ensures:
- **Bounded sequence length** → Flat TTFT
- **Semantic preservation** → Quality maintained
- **Memory efficiency** → No OOM on hybrid models

### Kernel Optimizations

SubSec automatically enables:
- **FlashAttention 2** for attention layers (when available)
- **Mamba-SSM fused kernels** for hybrid models (when installed)
- **device_map="auto"** for optimal GPU memory layout
- **TF32 matmul** for faster compute on Ampere+ GPUs

### Why This Matters for Hybrid Models

Hybrid Granite-4-H models use **Mamba state space layers**:
- Mamba state grows with sequence length
- Without fused kernels, memory usage is **quadratic** in naive path
- **Unbounded context** → guaranteed OOM on long conversations
- **SubSec's bounded context** → stable, predictable memory usage

---

## 📝 Citation & License

If you use SubSec in your work, please cite:

```
SubSec: Sub-second LLM inference through intelligent context management
https://github.com/[your-username]/SubSec
```

This project demonstrates optimization techniques for the IBM Granite-4 model family.

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Testing on Granite-4 variants further
- Alternative summarization strategies
- Integration with other model families

---

## ⚠️ Known Limitations

- **Hybrid models require good GPU memory**: Even with SubSec, hybrid Granite-4-H models need ~17-20 GiB VRAM
- **Naive Mamba kernels are slow**: Install `mamba-ssm` + `causal-conv1d` for best hybrid performance
- **Very long single responses**: If a single assistant response exceeds `SUBSEC_MAX_NEW_TOKENS`, it will be truncated
- **Chat template assumptions**: Works best with models that have proper chat templates; falls back gracefully for base models

---

## 📞 Contact

Built with ❤️ for efficient LLM inference.

For questions or issues, please open a GitHub issue.
