# Euclidean Geometric Attention — Parameter Golf Submission

**Fork of [openai/parameter-golf](https://github.com/openai/parameter-golf)**

## What is this?

A drop-in replacement for standard dot-product attention using **Euclidean geometric routing**. Instead of Q·K similarity, tokens are routed based on metric-weighted distance in a learned flat geometric space.

### Architecture: Euclidean Force Field Attention

Standard attention computes relevance via dot product: `softmax(Q·Kᵀ / √d)`.

We replace this with distance-based routing in Euclidean space:

```
centers = center_proj(x)           # WHERE each token sits (like Q)
metric  = softplus(metric_proj(x)) # force field strength (like K, always positive)
values  = value_proj(x)            # WHAT to communicate (V, unchanged)

dist²(i,j) = Σ_d (center_i - center_j)² × avg(metric_i, metric_j)_d
weights = softmax(-dist² × temperature)
output  = weights @ values
```

**Key properties:**
- **Same parameter count** as standard GQA attention (17,059,912 params) — zero overhead
- **Softplus metric** → always positive → space is truly flat Euclidean
- **k-NN is geometrically correct** — L2 distances are real distances, not approximate
- Supports GQA grouping, RoPE on centers, RMSNorm, tied embeddings — all baseline tricks preserved

### Why Euclidean?

Standard attention has no geometry — Q·K is a bilinear form with no spatial structure. Our model embeds tokens in a genuine metric space where:

1. **Distance means something** — nearby tokens are relevant, distant ones aren't
2. **k-NN retrieval is native** — you can sparsify attention by finding nearest neighbors in the geometric space, without any architectural changes
3. **The metric (force field) is learned per-token** — different tokens warp the local distance function differently

### Sub-Quadratic Inference (kNN Results)

Tested on a trained 13.6M param model (v8_vmlp, 200 epochs, OpenWebText):

| k (neighbors) | PPL | Δ vs full | Compute saved |
|---|---|---|---|
| full O(T²) | 126.4 | — | 0% |
| 128 | 127.2 | +0.6% | 50% |
| 64 | 131.5 | +4.0% | 75% |
| 32 | 139.5 | +10.3% | 87% |

**k=128 loses only 0.6% quality at half the compute.** A vanilla transformer has no space to do k-NN in — this capability is unique to geometric attention.

*Note: Competition eval uses full O(T²) attention. kNN is for deployment scaling.*

## Quick Start

```bash
# Clone this fork
git clone -b cfm-geometric-attention https://github.com/r2smith141/parameter-golf-CFM.git
cd parameter-golf-CFM

# Download FineWeb data
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (single GPU)
RUN_ID=euclidean_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=1 train_cfm.py

# Train (8×H100, competition setting)
RUN_ID=euclidean_8h100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_cfm.py
```

## Model Config

| Parameter | Value |
|-----------|-------|
| Layers | 9 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP multiplier | 2 (relu² MLP) |
| Vocab size | 1024 (SentencePiece BPE) |
| Tied embeddings | Yes |
| Total params | 17,059,912 |
| INT8+zlib size | ~4.7 MB (well under 16 MB limit) |

## File Structure

- `train_cfm.py` — Full training script with Euclidean geometric attention (adapted from baseline `train_gpt.py`)
- `train_gpt.py` — Original baseline (standard dot-product attention)
- `train_gpt_mlx.py` — Original MLX baseline for Apple Silicon
- `data/` — Dataset download scripts and tokenizers

## Background

This work comes from the **Dynamic Geometric Workspace (DGW)** research program exploring geometric inductive biases for language modeling. The Euclidean force field architecture (v8_vmlp) was developed and tested across 200+ training runs on models from 300K to 17M parameters.

## Authors

Richie Smith ([@r2smith141](https://github.com/r2smith141))
