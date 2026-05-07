"""
Logit lens analysis for v3 cognitive concept vectors.

For each centered concept vector v, compute lm_head.weight @ v to find which
output tokens v would up- or down-weight if added to the residual stream at
the unembedding layer. This is a sanity check that v's direction has
interpretable semantic content beyond the geometric structure we already
verified — the test "if I push the model in this direction, what words come
out?"

Anthropic's emotion paper (Table 1) is the reference: e.g., "happy" upweights
{excited, excitement, exciting, happ, celeb}, "sad" upweights {mour, grief,
tears, lonely, crying}.

The lm_head matrix is loaded directly from safetensors — no need to load the
full 35B model.

Usage:
    # Sanity: methodC, layer 30, top-15 (what you run first)
    python scripts/logit_lens_v3.py \\
        --vec-dir runs/cognitive_v3_full/extractions \\
        --output-dir outputs/cognitive_v3_full/analyses_methodC/logit_lens \\
        --sanity

    # Full: 4 methods x 4 layers
    python scripts/logit_lens_v3.py \\
        --vec-dir runs/cognitive_v3_full/extractions \\
        --output-dir outputs/cognitive_v3_full/analyses_methodC/logit_lens
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL_PATH = "/workspace/models/qwen3.6-35b-a3b-nf4"
LAYERS = [10, 20, 30, 36]
METHODS = ["methodA_v2style", "methodB_isolation", "methodC_incontext", "methodD_contrast"]


def get_lm_head_weight(model_path: Path, device: str = "cuda") -> torch.Tensor:
    """Load only lm_head.weight from safetensors directly to `device` in its native dtype.

    Qwen3-Next has vocab=248,320 — the bf16 weight is ~2.5GB and a fp32 numpy
    copy would push past container memory limits. Keep it on GPU in bf16.
    """
    from safetensors import safe_open

    model_path = Path(model_path)

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        candidates = [k for k in weight_map if k.endswith("lm_head.weight")]
        if not candidates:
            raise RuntimeError(
                f"lm_head.weight not in safetensors index. Available keys: "
                f"{list(weight_map.keys())[:8]}"
            )
        key = candidates[0]
        shard_file = model_path / weight_map[key]
        print(f"  loading {key} from {shard_file.name} -> {device}")
        with safe_open(shard_file, framework="pt", device=device) as f:
            W = f.get_tensor(key)
    else:
        single = model_path / "model.safetensors"
        if not single.exists():
            raise RuntimeError(f"No safetensors found in {model_path}")
        print(f"  loading lm_head.weight from {single.name} -> {device}")
        with safe_open(single, framework="pt", device=device) as f:
            keys = list(f.keys())
            candidates = [k for k in keys if k.endswith("lm_head.weight")]
            if not candidates:
                raise RuntimeError(f"lm_head.weight not found. Sample keys: {keys[:5]}")
            W = f.get_tensor(candidates[0])
    return W


def get_tokenizer(model_path: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_path), use_fast=False, trust_remote_code=True)


def logit_lens_top_k(W: torch.Tensor, vec: np.ndarray, top_k: int):
    """W is (vocab, hidden) on GPU. Returns (up_idx, down_idx, top_up_logits, top_down_logits)
    as small CPU arrays — never materializes the full (vocab,) logits on CPU."""
    v = torch.from_numpy(vec.astype(np.float32)).to(dtype=W.dtype, device=W.device)
    logits = W @ v  # (vocab,) on GPU in W.dtype
    up_vals, up_idx = torch.topk(logits, top_k, largest=True)
    down_vals, down_idx = torch.topk(logits, top_k, largest=False)
    return (up_idx.cpu().numpy(), down_idx.cpu().numpy(),
            up_vals.float().cpu().numpy(), down_vals.float().cpu().numpy())


def run_one(W, tok, vec_path: Path, top_k: int) -> dict:
    """Run logit lens for all concepts in one vector file. Returns dict per concept."""
    data = dict(np.load(vec_path))
    out = {}
    for concept, v in data.items():
        up_idx, down_idx, up_vals, down_vals = logit_lens_top_k(W, v, top_k)
        out[concept] = {
            "top_up": [
                {"id": int(i), "logit": round(float(lv), 4),
                 "tok": tok.decode([int(i)])}
                for i, lv in zip(up_idx, up_vals)
            ],
            "top_down": [
                {"id": int(i), "logit": round(float(lv), 4),
                 "tok": tok.decode([int(i)])}
                for i, lv in zip(down_idx, down_vals)
            ],
        }
    return out


def print_summary(entry: dict, label: str, k: int = 5):
    print(f"\n  === {label} ===")
    print(f"  Top-{k} UP tokens per concept:")
    for c, data in entry.items():
        toks = [t["tok"].strip().replace("\n", "\\n") for t in data["top_up"][:k]]
        print(f"    {c:<12} : {toks}")
    print(f"  Top-{k} DOWN tokens per concept:")
    for c, data in entry.items():
        toks = [t["tok"].strip().replace("\n", "\\n") for t in data["top_down"][:k]]
        print(f"    {c:<12} : {toks}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec-dir", required=True,
                    help="e.g. runs/cognitive_v3_full/extractions (with method subdirs)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--method", default=None,
                    help="Single method (e.g., methodC_incontext). Else all 4.")
    ap.add_argument("--layer", type=int, default=None,
                    help="Single layer. Else all of {10,20,30,36}.")
    ap.add_argument("--sanity", action="store_true",
                    help="Sanity mode: methodC_incontext @ layer_30 only")
    args = ap.parse_args()

    if args.sanity:
        methods, layers = ["methodC_incontext"], [30]
    else:
        methods = [args.method] if args.method else METHODS
        layers = [args.layer] if args.layer is not None else LAYERS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading lm_head from {args.model_path} (device={device})")
    W = get_lm_head_weight(Path(args.model_path), device=device)
    print(f"  lm_head shape: {tuple(W.shape)}, dtype: {W.dtype}, device: {W.device}")

    print(f"loading tokenizer ...")
    tok = get_tokenizer(Path(args.model_path))
    print(f"  vocab size: {tok.vocab_size}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    vec_root = Path(args.vec_dir)
    for method in methods:
        for layer in layers:
            vec_path = vec_root / method / f"layer_{layer}" / "concept_vectors_modeA.npz"
            if not vec_path.exists():
                print(f"\n[skip] {vec_path} not found")
                continue
            label = f"{method} @ layer_{layer}"
            print(f"\n[run] {label}")
            entry = run_one(W, tok, vec_path, args.top_k)
            summary[f"{method}/layer_{layer}"] = entry
            print_summary(entry, label, k=5)

    out_file = out_dir / ("logit_lens_sanity.json" if args.sanity else "logit_lens.json")
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_file} ({len(summary)} configs, "
          f"{sum(len(v) for v in summary.values())} total concept entries)")


if __name__ == "__main__":
    main()
