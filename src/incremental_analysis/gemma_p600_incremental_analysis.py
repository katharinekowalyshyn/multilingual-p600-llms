"""
Gemma P600 Incremental SAE Analysis

Runs per-token, position-aligned SAE feature analysis over control vs P600 datasets
using TransformerLens (Gemma) and SAELens (Neuronpedia SAE release).

Outputs:
- results/gemma_p600_pos_curve.png (per-position scalar curves)
- results/gemma_p600_pos_curve.csv (per-position scalars)
- results/gemma_p600_pos_feature_means.npz (optional per-position, per-feature means)

Reference notebook: https://colab.research.google.com/drive/1bJHQ6-BKpAAGGtDZQdzEwkvnTOEdViK2?usp=sharing
"""

from pathlib import Path
from typing import List, Tuple, Dict
import os
import math
import torch
import torch as t
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from saelens import SAE
sns.set_style("whitegrid")


# ----------------------
# Config
# ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------
# Data Loading
# ----------------------
def load_sentences(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    s = df["sentence"].astype(str).str.strip()
    return [x for x in s.tolist() if len(x) > 0]


def tokenize_sentences(sentences: List[str], model: HookedTransformer) -> List[t.Tensor]:
    tokens_list: List[t.Tensor] = []
    for text in sentences:
        toks = model.to_tokens(text, add_bos=True)
        # shape [1, seq_len]
        tokens_list.append(toks.squeeze(0))
    return tokens_list


def collate_pad(batch: List[t.Tensor], pad_id: int) -> Tuple[t.Tensor, t.Tensor]:
    # Right-pad to max length in the batch
    max_len = max(x.shape[-1] for x in batch)
    B = len(batch)
    out = t.full((B, max_len), pad_id, dtype=batch[0].dtype)
    mask = t.zeros((B, max_len), dtype=t.bool)
    for i, x in enumerate(batch):
        L = x.shape[-1]
        out[i, :L] = x
        mask[i, :L] = True
    return out, mask


# ----------------------
# SAE Hooking & Forward
# ----------------------
def run_dataset(tokens: List[t.Tensor], model: HookedTransformer, sae: SAE) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pos_scalar_means, pos_feature_means) where
    - pos_scalar_means: [max_pos] scalar mean across features per position
    - pos_feature_means: [max_pos, d_sae] mean across examples per position (may be large)
    """
    hook_point_name = sae.hook_point  # e.g., 'blocks.20.hook_resid_post'
    features_store: List[t.Tensor] = []  # list of [B, P, d_sae]

    def forward_hook(tensor: t.Tensor, hook: HookPoint):
        B, P, D = tensor.shape
        flat = tensor.view(B * P, D)
        feats = sae.encode(flat).view(B, P, -1)
        features_store.append(feats.detach().to("cpu"))

    model.add_hook(hook_point_name, forward_hook, dir="fwd")

    pad_id = model.tokenizer.pad_token_id
    # Some tokenizers use None for pad; use eos or 0 as safe fallback
    if pad_id is None:
        pad_id = getattr(model.tokenizer, "eos_token_id", 0) or 0

    loader = DataLoader(tokens, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_pad(b, pad_id))

    with t.inference_mode():
        for batch_tokens, batch_mask in tqdm(loader, desc=f"Forward {hook_point_name}"):
            batch_tokens = batch_tokens.to(DEVICE)
            _ = model(batch_tokens)

    # Remove hook
    model.reset_hooks(including_permanent=False)

    if not features_store:
        raise RuntimeError("No features captured; verify hook point and inputs.")

    # Concatenate along batch dimension
    feats_all = t.cat(features_store, dim=0)  # [N, P, d_sae]

    # Build a corresponding mask for all batches re-collated
    # We need masks for the same concatenation order
    # Re-run dataloader to build masks only (CPU, fast)
    all_masks: List[t.Tensor] = []
    for batch_tokens, batch_mask in DataLoader(tokens, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_pad(b, pad_id)):
        all_masks.append(batch_mask)
    mask_all = t.cat(all_masks, dim=0)  # [N, P]

    # Determine maximum position across all examples
    max_pos = feats_all.shape[1]
    d_sae = feats_all.shape[2]

    # Compute per-position feature means using masking
    # For each position p: take feats_all[:, p, :] over rows where mask_all[:, p] is True
    pos_feature_means = t.empty((max_pos, d_sae), dtype=t.float32)
    pos_scalar_means = t.empty((max_pos,), dtype=t.float32)

    for p in range(max_pos):
        valid = mask_all[:, p]
        if valid.any():
            feats_p = feats_all[valid, p, :]  # [n_valid, d_sae]
            mean_p = feats_p.mean(dim=0)
            pos_feature_means[p] = mean_p
            pos_scalar_means[p] = mean_p.mean()
        else:
            pos_feature_means[p] = 0.0
            pos_scalar_means[p] = 0.0

    return pos_scalar_means.numpy(), pos_feature_means.numpy()


def run_predictions(tokens: List[t.Tensor], model: HookedTransformer, label: str, results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute next-token predictions at each position and save detailed and per-position aggregates.

    Returns (df_long, df_pos) where:
    - df_long: columns [sentence_idx, position, input_token_id, input_token, target_token_id, target_token,
      pred_token_id, pred_token, pred_prob, true_prob, correct, dataset]
    - df_pos: per-position aggregates with columns [position, accuracy, mean_true_prob, mean_surprisal_bits, dataset]
    """
    tokenizer = model.tokenizer

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0

    # We need also sentence indices maintained across batching. Create list of indices.
    sentence_indices = list(range(len(tokens)))

    def collate_with_index(batch_items):
        # batch_items is list of (tokens, idx) tuples
        toks = [x[0] for x in batch_items]
        idxs = [x[1] for x in batch_items]
        padded, mask = collate_pad(toks, pad_id)
        return padded, mask, t.tensor(idxs, dtype=t.long)

    dataset = list(zip(tokens, sentence_indices))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_with_index)

    rows = []
    pos_stats: Dict[int, Dict[str, List[float]]] = {}

    with t.inference_mode():
        for batch_tokens, batch_mask, batch_idxs in tqdm(loader, desc=f"Predictions {label}"):
            batch_tokens = batch_tokens.to(DEVICE)
            logits = model(batch_tokens)  # [B, P, V]

            # Shift for next-token targets
            logits = logits[:, :-1, :]  # predict next token
            mask_next = batch_mask[:, 1:]  # True where next token exists
            targets = batch_tokens[:, 1:]  # [B, P-1]

            # Log-softmax for stable true prob and surprisal
            log_probs = t.log_softmax(logits, dim=-1)
            true_logprob = t.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [B, P-1]
            true_prob = true_logprob.exp()
            # Top-1 predictions
            pred_token_id = logits.argmax(dim=-1)  # [B, P-1]
            pred_prob = t.softmax(logits, dim=-1).gather(-1, pred_token_id.unsqueeze(-1)).squeeze(-1)
            correct = pred_token_id.eq(targets)

            B, Pm1 = pred_token_id.shape
            for i in range(B):
                sent_idx = int(batch_idxs[i].item())
                # input positions correspond to tokens up to second-to-last
                for p in range(Pm1):
                    if not bool(mask_next[i, p].item()):
                        continue
                    input_id = int(batch_tokens[i, p].item())
                    target_id = int(targets[i, p].item())
                    pred_id = int(pred_token_id[i, p].item())
                    rows.append({
                        "sentence_idx": sent_idx,
                        "position": p,
                        "input_token_id": input_id,
                        "input_token": tokenizer.decode([input_id], clean_up_tokenization_spaces=False),
                        "target_token_id": target_id,
                        "target_token": tokenizer.decode([target_id], clean_up_tokenization_spaces=False),
                        "pred_token_id": pred_id,
                        "pred_token": tokenizer.decode([pred_id], clean_up_tokenization_spaces=False),
                        "pred_prob": float(pred_prob[i, p].item()),
                        "true_prob": float(true_prob[i, p].item()),
                        "correct": bool(correct[i, p].item()),
                        "dataset": label,
                    })
                    st = pos_stats.setdefault(p, {"correct": [], "true_prob": []})
                    st["correct"].append(float(correct[i, p].item()))
                    st["true_prob"].append(float(true_prob[i, p].item()))

    df_long = pd.DataFrame(rows)
    pos_rows = []
    ln2 = math.log(2.0)
    for p, st in sorted(pos_stats.items()):
        arr_true = np.array(st["true_prob"], dtype=np.float64)
        arr_corr = np.array(st["correct"], dtype=np.float64)
        # Surprisal in bits: -log2(p)
        mean_surprisal_bits = float((-np.log(arr_true + 1e-12) / ln2).mean()) if arr_true.size else float("nan")
        pos_rows.append({
            "position": p,
            "accuracy": float(arr_corr.mean()) if arr_corr.size else float("nan"),
            "mean_true_prob": float(arr_true.mean()) if arr_true.size else float("nan"),
            "mean_surprisal_bits": mean_surprisal_bits,
            "dataset": label,
        })
    df_pos = pd.DataFrame(pos_rows)

    # Save
    long_path = results_dir / f"gemma_p600_predictions_{label}.csv"
    pos_path = results_dir / f"gemma_p600_predictions_{label}_per_pos.csv"
    df_long.to_csv(long_path, index=False)
    df_pos.to_csv(pos_path, index=False)
    print(f"Saved predictions: {long_path}")
    print(f"Saved per-position aggregates: {pos_path}")

    return df_long, df_pos


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    inc = cfg.get("incremental_p600")
    if not inc or not inc.get("enabled", True):
        print("incremental_p600 disabled in config.")
        return

    input_control = Path(hydra.utils.to_absolute_path(inc.input_files.control))
    input_p600 = Path(hydra.utils.to_absolute_path(inc.input_files.p600))
    output_dir = Path(hydra.utils.to_absolute_path(inc.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    gemma_checkpoint = inc.model.gemma_checkpoint
    sae_release_id = inc.model.sae_release_id
    batch_size = int(inc.batch_size)
    save_per_feature = bool(inc.save_per_feature)

    global BATCH_SIZE
    BATCH_SIZE = batch_size

    print(f"Using device: {DEVICE}")
    print(f"Gemma checkpoint: {gemma_checkpoint}")
    print(f"SAE release: {sae_release_id}")

    control_sents = load_sentences(input_control)
    p600_sents = load_sentences(input_p600)
    print(f"Loaded {len(control_sents)} control, {len(p600_sents)} P600 sentences")

    model = HookedTransformer.from_pretrained(
        gemma_checkpoint,
        device=DEVICE,
        center_writing_weights=False,
        center_unembed=False,
    )
    sae = SAE.from_pretrained(sae_release_id, device=DEVICE)
    print(f"Hook point: {sae.hook_point}; d_sae={sae.d_sae}")

    control_tokens = tokenize_sentences(control_sents, model)
    p600_tokens = tokenize_sentences(p600_sents, model)

    control_curve, control_pos_feat_means = run_dataset(control_tokens, model, sae)
    p600_curve, p600_pos_feat_means = run_dataset(p600_tokens, model, sae)

    max_pos = max(control_curve.shape[0], p600_curve.shape[0])
    if control_curve.shape[0] != max_pos:
        cc = np.full((max_pos,), np.nan, dtype=np.float32)
        cc[: control_curve.shape[0]] = control_curve
        control_curve = np.nan_to_num(cc, nan=0.0)
    if p600_curve.shape[0] != max_pos:
        pc = np.full((max_pos,), np.nan, dtype=np.float32)
        pc[: p600_curve.shape[0]] = p600_curve
        p600_curve = np.nan_to_num(pc, nan=0.0)

    diff_curve = p600_curve - control_curve

    plt.figure(figsize=(12, 6))
    plt.plot(control_curve, label="Control", alpha=0.8)
    plt.plot(p600_curve, label="P600", alpha=0.8)
    plt.plot(diff_curve, label="P600 - Control", alpha=0.8)
    plt.xlabel("Token position (0-index)")
    plt.ylabel("Mean SAE feature activation (scalar avg)")
    plt.title("Per-position SAE activation curves")
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / "gemma_p600_pos_curve.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig_path}")

    df = pd.DataFrame({
        "position": np.arange(max_pos),
        "control_mean": control_curve,
        "p600_mean": p600_curve,
        "difference": diff_curve,
    })
    csv_path = output_dir / "gemma_p600_pos_curve.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    if save_per_feature:
        npz_path = output_dir / "gemma_p600_pos_feature_means.npz"
        np.savez(
            npz_path,
            control_pos_feature_means=control_pos_feat_means,
            p600_pos_feature_means=p600_pos_feat_means,
        )
        print(f"Saved per-feature means: {npz_path}")

    print("Computing next-token predictions (control)...")
    _, control_pos = run_predictions(control_tokens, model, label="control", results_dir=output_dir)
    print("Computing next-token predictions (p600)...")
    _, p600_pos = run_predictions(p600_tokens, model, label="p600", results_dir=output_dir)

    merged = pd.merge(control_pos, p600_pos, on="position", suffixes=("_control", "_p600"))
    merged["surprisal_diff_bits"] = merged["mean_surprisal_bits_p600"] - merged["mean_surprisal_bits_control"]

    plt.figure(figsize=(12, 6))
    plt.plot(merged["position"], merged["mean_surprisal_bits_control"], label="Control surprisal (bits)")
    plt.plot(merged["position"], merged["mean_surprisal_bits_p600"], label="P600 surprisal (bits)")
    plt.plot(merged["position"], merged["surprisal_diff_bits"], label="P600 - Control (bits)")
    plt.xlabel("Token position (0-index)")
    plt.ylabel("Surprisal (bits)")
    plt.title("Per-position next-token surprisal")
    plt.legend()
    plt.tight_layout()
    fig2_path = output_dir / "gemma_p600_surprisal_pos_curve.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved figure: {fig2_path}")

    merged.to_csv(output_dir / "gemma_p600_surprisal_pos_curve.csv", index=False)
    print(f"Saved CSV: {output_dir / 'gemma_p600_surprisal_pos_curve.csv'}")

    print("Done.")


if __name__ == "__main__":
    main()


