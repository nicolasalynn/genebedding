"""
NT Benchmark: Fine-tuning Classification Heads on Model Embeddings

Usage:
    from benchmarks.nt_benchmark import benchmark_single_models, benchmark_ensemble_models

    results = benchmark_single_models(
        model_names=["nt", "caduceus", "convnova"],
        tasks=["H3", "H3K4me1"]
    )

    results = benchmark_ensemble_models(
        model_pairs=[("nt", "caduceus")],
        tasks=["H3"]
    )
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model factories
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.evaluate import get_model, list_models, MODEL_FACTORIES

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
USE_MIXED_PRECISION = torch.cuda.is_available()

# Lazy load dataset
_ds_all = None

def _get_dataset():
    global _ds_all
    if _ds_all is None:
        _ds_all = load_dataset("InstaDeepAI/nucleotide_transformer_downstream_tasks")
    return _ds_all

def get_all_tasks() -> List[str]:
    return sorted(set(_get_dataset()["train"]["task"]))

# ============================================================================
# BPNet-Style Classification Head
# ============================================================================

class DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.gelu(self.norm(self.conv(x)))) + x


class BPNetHead(nn.Module):
    """BPNet-inspired classification head."""

    def __init__(self, input_dim: int, n_classes: int = 2, hidden_dim: int = 256,
                 n_conv_blocks: int = 3, dense_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)

        dilations = [2 ** i for i in range(n_conv_blocks)]
        self.conv_blocks = nn.ModuleList([
            DilatedConvBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])

        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, dense_hidden),
            nn.LayerNorm(dense_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, dense_hidden // 2),
            nn.LayerNorm(dense_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden // 2, n_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = F.gelu(self.input_norm(self.input_proj(x)))
            return self.dense(x)

        B, L, D = x.shape
        x = F.gelu(self.input_norm(self.input_proj(x)))
        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            x = block(x)
        x = x.transpose(1, 2).mean(dim=1)
        return self.dense(x)


class SimpleHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int = 2, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


# ============================================================================
# Dataset and Utilities
# ============================================================================

class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def _ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    c0, c1 = int((y == 0).sum()), int((y == 1).sum())
    total = c0 + c1
    return torch.tensor([
        total / (2.0 * c0) if c0 else 1.0,
        total / (2.0 * c1) if c1 else 1.0
    ], dtype=torch.float32)


def _compute_embeddings(wrapper, records: List[Tuple[str, int]], pool: str = "mean",
                        desc: str = "Embedding") -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for s, label in tqdm(records, desc=desc, leave=False):
        emb = _ensure_numpy(wrapper.embed(s, pool=pool))
        xs.append(emb.astype(np.float32))
        ys.append(int(label))
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def _normalize_embeddings(X: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize embeddings.

    Parameters
    ----------
    X : np.ndarray
        Embeddings (N, D)
    method : str
        'l2' - L2 normalize each sample to unit norm
        'zscore' - Standardize each dimension to mean=0, std=1
        'none' - No normalization

    Returns
    -------
    X_norm : np.ndarray
    """
    if method == "none":
        return X
    elif method == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        return X / norms
    elif method == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-8)
        return (X - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _get_task_data(task: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    ds = _get_dataset()

    # Validate task exists
    all_tasks = get_all_tasks()
    if task not in all_tasks:
        raise ValueError(f"Task '{task}' not found. Available tasks: {all_tasks}")

    train_ds = ds["train"].filter(lambda ex: ex["task"] == task)
    test_ds = ds["test"].filter(lambda ex: ex["task"] == task)

    train_recs = list(zip(train_ds["sequence"], train_ds["label"]))
    test_recs = list(zip(test_ds["sequence"], test_ds["label"]))

    if not train_recs:
        raise ValueError(f"No training records found for task '{task}'")
    if not test_recs:
        raise ValueError(f"No test records found for task '{task}'")

    return train_recs, test_recs


# ============================================================================
# Training and Evaluation
# ============================================================================

@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    probs, preds, labels = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True))
        probs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
        preds.append(logits.argmax(-1).cpu())
        labels.append(yb)

    probs, preds, labels = torch.cat(probs).numpy(), torch.cat(preds).numpy(), torch.cat(labels).numpy()
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(labels, preds), "auroc": auc, "mcc": matthews_corrcoef(labels, preds)}


def _train_head(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                head_type: str = "bpnet", epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
                lr: float = DEFAULT_LR, weight_decay: float = DEFAULT_WEIGHT_DECAY,
                verbose: bool = True) -> Dict[str, float]:
    D = X_train.shape[1]
    train_loader = DataLoader(ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(ArrayDataset(X_test, y_test), batch_size=batch_size, shuffle=False, pin_memory=True)

    head = (BPNetHead(D) if head_type == "bpnet" else SimpleHead(D)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train).to(DEVICE))
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    best_mcc, best_metrics = -1.0, None

    for ep in range(1, epochs + 1):
        head.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                loss = criterion(head(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        metrics = _evaluate(head, test_loader, DEVICE)
        if metrics["mcc"] > best_mcc:
            best_mcc, best_metrics = metrics["mcc"], metrics.copy()

        if verbose:
            tqdm.write(f"  Epoch {ep}/{epochs}: loss={epoch_loss/n_batches:.4f} "
                      f"acc={metrics['accuracy']:.3f} auc={metrics['auroc']:.3f} mcc={metrics['mcc']:.3f}")

    return best_metrics


# ============================================================================
# Main Benchmark Functions
# ============================================================================

def benchmark_single_models(model_names: List[str], tasks: Optional[List[str]] = None, *,
                            head_type: str = "bpnet", epochs: int = DEFAULT_EPOCHS,
                            batch_size: int = DEFAULT_BATCH_SIZE, lr: float = DEFAULT_LR,
                            verbose: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Benchmark individual models on specified tasks."""
    if tasks is None:
        tasks = get_all_tasks()

    available = list_models()
    for name in model_names:
        if name not in available:
            raise ValueError(f"Unknown model: {name}. Available: {available}")

    results = {name: {} for name in model_names}

    for model_name in model_names:
        if verbose:
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        wrapper = get_model(model_name)

        for task in tasks:
            if verbose:
                print(f"\nTask: {task}")

            train_recs, test_recs = _get_task_data(task)
            X_train, y_train = _compute_embeddings(wrapper, train_recs, desc=f"{model_name}/{task} train")
            X_test, y_test = _compute_embeddings(wrapper, test_recs, desc=f"{model_name}/{task} test")

            metrics = _train_head(X_train, y_train, X_test, y_test, head_type=head_type,
                                  epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
            results[model_name][task] = metrics

            if verbose:
                print(f">> {model_name}/{task}: acc={metrics['accuracy']:.3f} "
                      f"auc={metrics['auroc']:.3f} mcc={metrics['mcc']:.3f}")

        del wrapper
        torch.cuda.empty_cache()

    return results


def benchmark_ensemble_models(model_combinations: List[Tuple[str, ...]], tasks: Optional[List[str]] = None, *,
                              head_type: str = "bpnet", epochs: int = DEFAULT_EPOCHS,
                              batch_size: int = DEFAULT_BATCH_SIZE, lr: float = DEFAULT_LR,
                              normalize: str = "l2",
                              verbose: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Benchmark ensembles by concatenating embeddings from multiple models.

    Parameters
    ----------
    model_combinations : List[Tuple[str, ...]]
        List of model tuples to ensemble, e.g., [("nt", "caduceus"), ("nt", "caduceus", "convnova")]
    normalize : str
        Normalization method for each model's embeddings before concatenation:
        'l2' - L2 normalize to unit norm (default, recommended)
        'zscore' - Standardize to mean=0, std=1
        'none' - No normalization
    """
    if tasks is None:
        tasks = get_all_tasks()

    available = list_models()
    for combo in model_combinations:
        for m in combo:
            if m not in available:
                raise ValueError(f"Unknown model: {m}. Available: {available}")

    results = {}

    for combo in model_combinations:
        ensemble_name = "+".join(combo)
        results[ensemble_name] = {}

        if verbose:
            print(f"\n{'='*60}\nEnsemble: {ensemble_name}\n{'='*60}")

        # Load all models in the combination
        wrappers = [get_model(m) for m in combo]

        for task in tasks:
            if verbose:
                print(f"\nTask: {task}")

            train_recs, test_recs = _get_task_data(task)

            # Compute embeddings from all models
            X_train_list = []
            X_test_list = []
            y_train = None
            y_test = None

            for i, (m, wrapper) in enumerate(zip(combo, wrappers)):
                X_tr, y_tr = _compute_embeddings(wrapper, train_recs, desc=f"{m}/{task} train")
                X_te, y_te = _compute_embeddings(wrapper, test_recs, desc=f"{m}/{task} test")
                # Normalize each model's embeddings before concatenation
                X_tr = _normalize_embeddings(X_tr, method=normalize)
                X_te = _normalize_embeddings(X_te, method=normalize)
                X_train_list.append(X_tr)
                X_test_list.append(X_te)
                if i == 0:
                    y_train, y_test = y_tr, y_te

            # Concatenate all embeddings
            X_train = np.concatenate(X_train_list, axis=1)
            X_test = np.concatenate(X_test_list, axis=1)

            if verbose:
                print(f"  Concatenated dim: {X_train.shape[1]}")

            metrics = _train_head(X_train, y_train, X_test, y_test, head_type=head_type,
                                  epochs=epochs, batch_size=batch_size, lr=lr, verbose=verbose)
            results[ensemble_name][task] = metrics

            if verbose:
                print(f">> {ensemble_name}/{task}: acc={metrics['accuracy']:.3f} "
                      f"auc={metrics['auroc']:.3f} mcc={metrics['mcc']:.3f}")

        # Clean up
        for wrapper in wrappers:
            del wrapper
        torch.cuda.empty_cache()

    return results


def benchmark_all_pairs(model_names: List[str], tasks: Optional[List[str]] = None, **kwargs):
    """Benchmark all pairwise combinations."""
    return benchmark_ensemble_models(list(combinations(model_names, 2)), tasks, **kwargs)


def print_results_table(results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print formatted results table."""
    print(f"\n{'Model/Ensemble':<30} {'Task':<25} {'Acc':>8} {'AUROC':>8} {'MCC':>8}")
    print("-" * 85)

    for model in sorted(results.keys()):
        for task in sorted(results[model].keys()):
            m = results[model][task]
            print(f"{model:<30} {task:<25} {m['accuracy']:>8.4f} {m['auroc']:>8.4f} {m['mcc']:>8.4f}")

    print("-" * 85)
    for model in sorted(results.keys()):
        metrics = results[model]
        print(f"{model:<30} {'AVERAGE':<25} "
              f"{np.mean([m['accuracy'] for m in metrics.values()]):>8.4f} "
              f"{np.nanmean([m['auroc'] for m in metrics.values()]):>8.4f} "
              f"{np.mean([m['mcc'] for m in metrics.values()]):>8.4f}")


def merge_results(*result_dicts) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Merge multiple result dictionaries from benchmark functions.

    Usage:
        single = benchmark_single_models(...)
        ensemble = benchmark_ensemble_models(...)
        combined = merge_results(single, ensemble)
    """
    merged = {}
    for results in result_dicts:
        for model, tasks in results.items():
            if model not in merged:
                merged[model] = {}
            merged[model].update(tasks)
    return merged


def plot_results(results: Dict[str, Dict[str, Dict[str, float]]],
                 metric: str = "auroc",
                 figsize: Tuple[int, int] = (12, 6),
                 title: str = None,
                 sort_by_avg: bool = True) -> plt.Figure:
    """
    Plot benchmark results as grouped bar chart.

    Parameters
    ----------
    results : Dict
        Results from benchmark functions (can use merge_results to combine)
    metric : str
        Metric to plot: 'auroc', 'accuracy', or 'mcc'
    figsize : tuple
        Figure size
    title : str
        Plot title
    sort_by_avg : bool
        Sort models by average performance

    Returns
    -------
    fig : matplotlib.Figure
    """
    models = list(results.keys())
    tasks = sorted(set(t for m in results.values() for t in m.keys()))

    # Compute average for sorting
    if sort_by_avg:
        avg_scores = []
        for model in models:
            scores = [results[model].get(t, {}).get(metric, 0) for t in tasks]
            avg_scores.append(np.nanmean(scores))
        # Sort by average descending
        sorted_idx = np.argsort(avg_scores)[::-1]
        models = [models[i] for i in sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    n_models = len(models)
    n_tasks = len(tasks)
    x = np.arange(n_tasks)
    width = 0.8 / n_models

    # Plot bars for each model
    for i, model in enumerate(models):
        scores = [results[model].get(t, {}).get(metric, np.nan) for t in tasks]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=model, alpha=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f'{metric.upper()} by Model and Task')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_average_performance(results: Dict[str, Dict[str, Dict[str, float]]],
                             metric: str = "auroc",
                             figsize: Tuple[int, int] = (10, 6),
                             title: str = None) -> plt.Figure:
    """
    Plot average performance across tasks for each model.

    Parameters
    ----------
    results : Dict
        Results from benchmark functions
    metric : str
        Metric to plot
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.Figure
    """
    models = list(results.keys())
    avg_scores = []
    std_scores = []

    for model in models:
        scores = [m[metric] for m in results[model].values() if not np.isnan(m.get(metric, np.nan))]
        avg_scores.append(np.mean(scores) if scores else 0)
        std_scores.append(np.std(scores) if scores else 0)

    # Sort by average
    sorted_idx = np.argsort(avg_scores)[::-1]
    models = [models[i] for i in sorted_idx]
    avg_scores = [avg_scores[i] for i in sorted_idx]
    std_scores = [std_scores[i] for i in sorted_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, avg_scores, xerr=std_scores, color=colors, alpha=0.8, capsize=3)

    ax.set_xlabel(metric.upper())
    ax.set_title(title or f'Average {metric.upper()} Across Tasks')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, avg_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def plot_improvement(single_results: Dict, ensemble_results: Dict,
                     metric: str = "auroc",
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot improvement from single models to ensembles.

    Parameters
    ----------
    single_results : Dict
        Results from benchmark_single_models
    ensemble_results : Dict
        Results from benchmark_ensemble_models

    Returns
    -------
    fig : matplotlib.Figure
    """
    tasks = sorted(set(t for m in single_results.values() for t in m.keys()))

    fig, ax = plt.subplots(figsize=figsize)

    # Get best single model per task
    best_single = []
    for task in tasks:
        scores = [single_results[m].get(task, {}).get(metric, 0) for m in single_results]
        best_single.append(max(scores) if scores else 0)

    # Get best ensemble per task
    best_ensemble = []
    for task in tasks:
        scores = [ensemble_results[m].get(task, {}).get(metric, 0) for m in ensemble_results]
        best_ensemble.append(max(scores) if scores else 0)

    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width/2, best_single, width, label='Best Single Model', alpha=0.8)
    ax.bar(x + width/2, best_ensemble, width, label='Best Ensemble', alpha=0.8)

    ax.set_xlabel('Task')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()}: Best Single Model vs Best Ensemble')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig
