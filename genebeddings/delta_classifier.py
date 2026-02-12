"""
Trainable classification head on delta embeddings (mutant − wildtype).

Provides :class:`DeltaClassifier`, a PyTorch MLP that learns to classify
mutations by their effect in embedding space.  Supports:

* Loading deltas from :class:`VariantEmbeddingDB` or computing on-the-fly
* Preprocessing: standardization, PCA whitening, or L2 normalization
* Automatic class-weight balancing for imbalanced groups
* Early stopping with validation split
* k-fold cross-validation evaluation
* Save / load of trained models
* Auto-detection of single-variant vs epistasis IDs
"""
from __future__ import annotations

import copy
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
_DEFAULT_CONTEXT = 3000
_DEFAULT_GENOME = "hg38"

# ------------------------------------------------------------------ #
#  Result dataclasses
# ------------------------------------------------------------------ #

@dataclass
class TrainResult:
    """Returned by :meth:`DeltaClassifier.fit`."""
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: List[float]
    val_losses: List[float]
    train_accs: List[float]
    val_accs: List[float]


@dataclass
class DeltaClassificationResult:
    """Returned by :meth:`DeltaClassifier.classify`."""
    best: str
    confidence: float
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"best": self.best, "confidence": self.confidence}
        for k, v in self.scores.items():
            d[f"score_{k}"] = v
        return d

    def __repr__(self) -> str:
        lines = [f"DeltaClassificationResult(best={self.best!r}, confidence={self.confidence:.3f})"]
        for k, v in self.scores.items():
            lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)


@dataclass
class DeltaValidationResult:
    """Returned by :meth:`DeltaClassifier.validate`."""
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray
    labels: List[str]
    cv_accuracy: Optional[float] = None
    cv_std: Optional[float] = None
    cv_per_fold: Optional[List[float]] = None

    def __repr__(self) -> str:
        lines = [f"DeltaValidationResult(accuracy={self.accuracy:.4f})"]
        if self.cv_accuracy is not None:
            lines[0] = (
                f"DeltaValidationResult(accuracy={self.accuracy:.4f}, "
                f"cv={self.cv_accuracy:.4f}\u00b1{self.cv_std:.4f})"
            )
        for k, v in self.per_class_accuracy.items():
            lines.append(f"  {k}: {v:.1%}")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  PyTorch MLP
# ------------------------------------------------------------------ #

def _build_mlp(
    input_dim: int,
    num_classes: int,
    hidden_dims: Sequence[int] = (256, 64),
    dropout: float = 0.3,
    batch_norm: bool = True,
):
    """Build a simple MLP classification head (lazy import of torch)."""
    import torch.nn as nn

    layers: list = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ #
#  Preprocessing helpers
# ------------------------------------------------------------------ #

class _Preprocessor:
    """
    Fit-once, apply-many preprocessing for delta vectors.

    Modes
    -----
    ``"none"``
        No transformation.
    ``"l2"``
        L2-normalize each vector (unit length). Discards magnitude.
    ``"standardize"``
        Per-dimension z-score (subtract mean, divide by std).
    ``"whiten"``
        PCA whitening: project onto PCA axes, scale by 1/sqrt(eigenvalue).
        Optionally truncates to *whiten_k* components (or keeps components
        explaining *whiten_variance* fraction of variance).  This is
        equivalent to multiplying by ``cov^{-1/2}`` in the retained
        subspace, making Euclidean distances = Mahalanobis distances.
    """

    def __init__(
        self,
        mode: str = "none",
        whiten_k: Optional[int] = None,
        whiten_variance: Optional[float] = 0.99,
        eps: float = 1e-8,
    ):
        if mode not in ("none", "l2", "standardize", "whiten"):
            raise ValueError(
                f"Unknown preprocessing mode {mode!r}. "
                f"Choose from 'none', 'l2', 'standardize', 'whiten'."
            )
        self.mode = mode
        self.whiten_k = whiten_k
        self.whiten_variance = whiten_variance
        self.eps = eps

        # Fitted parameters
        self.mean_: Optional[np.ndarray] = None   # (D,)
        self.std_: Optional[np.ndarray] = None     # (D,) for standardize
        self.components_: Optional[np.ndarray] = None  # (k, D) for whiten
        self.scale_: Optional[np.ndarray] = None       # (k,) = 1/sqrt(eigenvalue)
        self.output_dim_: Optional[int] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "_Preprocessor":
        """Fit preprocessing parameters from training data (N x D)."""
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if self.mode == "none" or self.mode == "l2":
            self.output_dim_ = d

        elif self.mode == "standardize":
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ < self.eps] = 1.0  # avoid div-by-zero
            self.output_dim_ = d

        elif self.mode == "whiten":
            self.mean_ = X.mean(axis=0)
            Xc = X - self.mean_

            # SVD (more stable than eigh on covariance)
            # Xc = U @ S @ Vt, cov = Vt.T @ diag(S^2/(n-1)) @ Vt
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            eigenvalues = (S ** 2) / max(n - 1, 1)

            # Determine number of components to keep
            total_var = eigenvalues.sum()
            if self.whiten_k is not None:
                k = min(self.whiten_k, len(eigenvalues))
            elif self.whiten_variance is not None:
                cumvar = np.cumsum(eigenvalues) / (total_var + self.eps)
                k = int(np.searchsorted(cumvar, self.whiten_variance) + 1)
                k = min(k, len(eigenvalues))
            else:
                k = len(eigenvalues)

            # Truncate noisy components with tiny eigenvalues
            good = eigenvalues[:k] > self.eps
            k = int(good.sum()) if good.sum() > 0 else 1

            self.components_ = Vt[:k]               # (k, D)
            self.scale_ = 1.0 / np.sqrt(eigenvalues[:k] + self.eps)  # (k,)
            self.output_dim_ = k

            logger.info(
                "Whitening: %d -> %d dims (%.1f%% variance retained)",
                d, k, 100 * eigenvalues[:k].sum() / (total_var + self.eps),
            )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted preprocessing to data (N x D) or (D,)."""
        squeeze = X.ndim == 1
        if squeeze:
            X = X[np.newaxis, :]
        X = np.asarray(X, dtype=np.float32)

        if self.mode == "none":
            pass

        elif self.mode == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True) + self.eps
            X = X / norms

        elif self.mode == "standardize":
            X = (X - self.mean_.astype(np.float32)) / self.std_.astype(np.float32)

        elif self.mode == "whiten":
            Xc = X - self.mean_.astype(np.float32)
            # Project and scale: X_white = (Xc @ V.T) * scale
            X = (Xc @ self.components_.T.astype(np.float32)) * self.scale_.astype(np.float32)

        if squeeze:
            X = X[0]
        return X

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "whiten_k": self.whiten_k,
            "whiten_variance": self.whiten_variance,
            "eps": self.eps,
            "mean_": self.mean_,
            "std_": self.std_,
            "components_": self.components_,
            "scale_": self.scale_,
            "output_dim_": self.output_dim_,
            "fitted": self._fitted,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "_Preprocessor":
        obj = cls(
            mode=d["mode"],
            whiten_k=d.get("whiten_k"),
            whiten_variance=d.get("whiten_variance"),
            eps=d.get("eps", 1e-8),
        )
        obj.mean_ = d.get("mean_")
        obj.std_ = d.get("std_")
        obj.components_ = d.get("components_")
        obj.scale_ = d.get("scale_")
        obj.output_dim_ = d.get("output_dim_")
        obj._fitted = d.get("fitted", False)
        return obj


# ------------------------------------------------------------------ #
#  Delta loading (reuse helpers from mechanism_signatures)
# ------------------------------------------------------------------ #

def _get_delta_auto(
    mut_id: str,
    db=None,
    model=None,
    which: str = "M1",
    *,
    context: int = _DEFAULT_CONTEXT,
    genome: str = _DEFAULT_GENOME,
    pool: str = "mean",
    save_to_db: bool = True,
) -> Optional[np.ndarray]:
    """Delegate to mechanism_signatures._get_delta_auto."""
    from .mechanism_signatures import _get_delta_auto as _gda
    return _gda(
        mut_id, db=db, model=model, which=which,
        context=context, genome=genome, pool=pool,
        save_to_db=save_to_db,
    )


# ------------------------------------------------------------------ #
#  Main classifier
# ------------------------------------------------------------------ #

class DeltaClassifier:
    """
    Trainable MLP classification head on delta embeddings.

    Typical workflow::

        clf = DeltaClassifier.from_mutation_groups(
            {"splicing": [...], "missense": [...]},
            db=db,
            model=model,
            preprocessing="whiten",
        )
        result = clf.fit(epochs=100)
        val = clf.validate(cv=5)
        pred = clf.classify_variant("GENE:CHR:POS:REF:ALT", db=db)

    Parameters
    ----------
    input_dim : int
        Dimensionality of the delta vectors (before preprocessing).
    labels : list of str
        Ordered class labels.
    hidden_dims : tuple of int
        Hidden layer sizes for the MLP.
    dropout : float
        Dropout probability.
    batch_norm : bool
        Whether to use batch normalisation.
    preprocessing : str
        ``"none"``, ``"l2"``, ``"standardize"``, or ``"whiten"``.
    whiten_k : int or None
        If preprocessing is ``"whiten"``, fix the number of PCA components.
    whiten_variance : float or None
        If preprocessing is ``"whiten"`` and *whiten_k* is None, keep
        enough components to explain this fraction of variance (default 0.99).
    context, genome, pool : str
        Defaults for on-the-fly embedding.
    """

    def __init__(
        self,
        input_dim: int,
        labels: List[str],
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        preprocessing: str = "none",
        whiten_k: Optional[int] = None,
        whiten_variance: Optional[float] = 0.99,
        context: int = _DEFAULT_CONTEXT,
        genome: str = _DEFAULT_GENOME,
        pool: str = "mean",
    ):
        import torch
        self.raw_input_dim = input_dim
        self.labels = list(labels)
        self.label2idx = {l: i for i, l in enumerate(self.labels)}
        self.num_classes = len(labels)
        self.context = context
        self.genome = genome
        self.pool = pool

        self._hidden_dims = tuple(hidden_dims)
        self._dropout = dropout
        self._batch_norm = batch_norm

        # Preprocessing
        self.preprocessor = _Preprocessor(
            mode=preprocessing,
            whiten_k=whiten_k,
            whiten_variance=whiten_variance,
        )

        # MLP will be (re-)built once preprocessing is fitted
        # (because whiten can change the input dim)
        self._effective_input_dim = input_dim
        self.net = _build_mlp(
            input_dim, self.num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
        )
        self.device = torch.device("cpu")

        # Training data stored for CV / retraining
        self._X: Optional[np.ndarray] = None   # raw (before preprocessing)
        self._y: Optional[np.ndarray] = None
        self._group_ids: Optional[Dict[str, List[str]]] = None
        self._fitted = False

    def _rebuild_net(self, effective_dim: int) -> None:
        """Rebuild the MLP if preprocessing changes the input dimension."""
        if effective_dim != self._effective_input_dim:
            self._effective_input_dim = effective_dim
            self.net = _build_mlp(
                effective_dim, self.num_classes,
                hidden_dims=self._hidden_dims,
                dropout=self._dropout,
                batch_norm=self._batch_norm,
            ).to(self.device)

    # ------------------------------------------------------------------ #
    #  Construction from labelled mutation groups
    # ------------------------------------------------------------------ #

    @classmethod
    def from_mutation_groups(
        cls,
        groups: Dict[str, List[str]],
        db=None,
        model=None,
        which: str = "M1",
        *,
        hidden_dims: Sequence[int] = (256, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        preprocessing: str = "none",
        whiten_k: Optional[int] = None,
        whiten_variance: Optional[float] = 0.99,
        context: int = _DEFAULT_CONTEXT,
        genome: str = _DEFAULT_GENOME,
        pool: str = "mean",
        save_to_db: bool = True,
        show_progress: bool = False,
    ) -> "DeltaClassifier":
        """
        Build a classifier from ``{label: [mut_ids]}`` dictionary.

        Loads (or computes) delta vectors for every mutation ID,
        instantiates the MLP, and stores the training data so that
        :meth:`fit` can be called afterwards.
        """
        all_deltas: List[np.ndarray] = []
        all_labels: List[int] = []
        group_ids: Dict[str, List[str]] = {}
        label_order = sorted(groups.keys())

        for label_idx, label in enumerate(label_order):
            mut_ids = groups[label]
            group_ids[label] = []
            iterator: Iterable = mut_ids
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(mut_ids, desc=label)
                except ImportError:
                    pass

            for mid in iterator:
                delta = _get_delta_auto(
                    mid, db=db, model=model, which=which,
                    context=context, genome=genome, pool=pool,
                    save_to_db=save_to_db,
                )
                if delta is None:
                    logger.warning("Skipping %s (could not load/compute)", mid)
                    continue
                all_deltas.append(delta.flatten())
                all_labels.append(label_idx)
                group_ids[label].append(mid)

        if not all_deltas:
            raise ValueError("No deltas could be loaded for any group")

        X = np.stack(all_deltas, axis=0).astype(np.float32)
        y = np.array(all_labels, dtype=np.int64)

        logger.info(
            "Loaded %d deltas across %d classes (dim=%d): %s",
            len(X), len(label_order), X.shape[1],
            {l: int((y == i).sum()) for i, l in enumerate(label_order)},
        )

        obj = cls(
            input_dim=X.shape[1],
            labels=label_order,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            preprocessing=preprocessing,
            whiten_k=whiten_k,
            whiten_variance=whiten_variance,
            context=context,
            genome=genome,
            pool=pool,
        )
        obj._X = X
        obj._y = y
        obj._group_ids = group_ids
        return obj

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #

    def to(self, device) -> "DeltaClassifier":
        """Move model to a device (e.g. ``'cuda'``)."""
        import torch
        self.device = torch.device(device)
        self.net = self.net.to(self.device)
        return self

    def fit(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        val_fraction: float = 0.15,
        patience: int = 20,
        class_weight: Union[str, None, Dict[str, float]] = "balanced",
        scheduler: str = "cosine",
        seed: int = 42,
        verbose: bool = True,
    ) -> TrainResult:
        """
        Train the MLP on the loaded deltas.

        Parameters
        ----------
        epochs : int
            Maximum number of training epochs.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularisation.
        batch_size : int
            Mini-batch size.
        val_fraction : float
            Fraction of data held out for validation / early stopping.
            Set to 0 to train on everything (no early stopping).
        patience : int
            Stop if validation loss hasn't improved for this many epochs.
        class_weight : "balanced", None, or dict
            ``"balanced"`` computes inverse-frequency weights.
            A dict maps label -> weight.  ``None`` uses uniform weights.
        scheduler : str
            ``"cosine"`` or ``"none"``.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Print progress.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self._X is None:
            raise RuntimeError("No training data. Call from_mutation_groups first.")

        rng = np.random.RandomState(seed)
        X_raw, y = self._X.copy(), self._y.copy()

        # Train / val split (split BEFORE fitting preprocessing to avoid data leakage)
        n = len(X_raw)
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction)) if val_fraction > 0 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        # Fit preprocessing on training split only
        self.preprocessor.fit(X_raw[train_idx])
        self._rebuild_net(self.preprocessor.output_dim_)

        # Apply preprocessing
        X = self.preprocessor.transform(X_raw)

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        if n_val > 0:
            X_val = torch.tensor(X[val_idx], dtype=torch.float32)
            y_val = torch.tensor(y[val_idx], dtype=torch.long)

        # Class weights
        if class_weight == "balanced":
            counts = np.bincount(y[train_idx], minlength=self.num_classes).astype(np.float64)
            counts = np.maximum(counts, 1)
            w = (1.0 / counts) * len(y[train_idx]) / self.num_classes
            weight_tensor = torch.tensor(w, dtype=torch.float32).to(self.device)
        elif isinstance(class_weight, dict):
            w = np.ones(self.num_classes, dtype=np.float64)
            for label, wt in class_weight.items():
                w[self.label2idx[label]] = wt
            weight_tensor = torch.tensor(w, dtype=torch.float32).to(self.device)
        else:
            weight_tensor = None

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay,
        )

        if scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            sched = None

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        # Move to device
        self.net = self.net.to(self.device)

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(self.net.state_dict())
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, epochs + 1):
            # --- train ---
            self.net.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(xb)
            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            if sched is not None:
                sched.step()

            # --- val ---
            if n_val > 0:
                self.net.eval()
                with torch.no_grad():
                    xv = X_val.to(self.device)
                    yv = y_val.to(self.device)
                    logits_v = self.net(xv)
                    vloss = criterion(logits_v, yv).item()
                    vacc = (logits_v.argmax(1) == yv).float().mean().item()
                val_losses.append(vloss)
                val_accs.append(vacc)

                if vloss < best_val_loss:
                    best_val_loss = vloss
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.net.state_dict())
                elif epoch - best_epoch >= patience:
                    if verbose:
                        logger.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                    break

                if verbose and epoch % max(1, epochs // 20) == 0:
                    logger.info(
                        "Epoch %3d  train_loss=%.4f  train_acc=%.3f  "
                        "val_loss=%.4f  val_acc=%.3f",
                        epoch, train_loss, train_acc, vloss, vacc,
                    )
            else:
                val_losses.append(train_loss)
                val_accs.append(train_acc)
                best_val_loss = train_loss
                best_epoch = epoch
                best_state = copy.deepcopy(self.net.state_dict())
                if verbose and epoch % max(1, epochs // 20) == 0:
                    logger.info(
                        "Epoch %3d  train_loss=%.4f  train_acc=%.3f",
                        epoch, train_loss, train_acc,
                    )

        # Restore best weights
        self.net.load_state_dict(best_state)
        self.net.eval()
        self._fitted = True

        if verbose:
            logger.info(
                "Training done. Best epoch=%d, best_val_loss=%.4f",
                best_epoch, best_val_loss,
            )

        return TrainResult(
            epochs_run=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
        )

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def _prepare_delta(self, delta: np.ndarray) -> "torch.Tensor":
        import torch
        v = np.asarray(delta, dtype=np.float32).flatten()
        v = self.preprocessor.transform(v)
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device)

    def classify(self, delta: np.ndarray) -> DeltaClassificationResult:
        """
        Classify a single delta vector.
        """
        import torch
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        self.net.eval()
        with torch.no_grad():
            x = self._prepare_delta(delta)
            logits = self.net(x).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

        best_idx = int(np.argmax(probs))
        scores = {self.labels[i]: float(probs[i]) for i in range(self.num_classes)}
        return DeltaClassificationResult(
            best=self.labels[best_idx],
            confidence=float(probs[best_idx]) * 100,
            scores=scores,
        )

    def classify_variant(
        self,
        mut_id: str,
        db=None,
        model=None,
        which: str = "M1",
    ) -> DeltaClassificationResult:
        """
        Load/compute delta for *mut_id* and classify.

        Auto-detects epistasis IDs (containing ``|``).
        """
        delta = _get_delta_auto(
            mut_id, db=db, model=model, which=which,
            context=self.context, genome=self.genome, pool=self.pool,
        )
        if delta is None:
            raise ValueError(f"Could not load or compute delta for {mut_id!r}")
        return self.classify(delta)

    def classify_batch(
        self,
        deltas: np.ndarray,
    ) -> List[DeltaClassificationResult]:
        """
        Classify a batch of delta vectors (N x D array).
        """
        import torch
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = np.asarray(deltas, dtype=np.float32)
        X = self.preprocessor.transform(X)

        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        results = []
        for i in range(len(probs)):
            best_idx = int(np.argmax(probs[i]))
            scores = {self.labels[j]: float(probs[i, j]) for j in range(self.num_classes)}
            results.append(DeltaClassificationResult(
                best=self.labels[best_idx],
                confidence=float(probs[i, best_idx]) * 100,
                scores=scores,
            ))
        return results

    def classify_variants(
        self,
        mut_ids: List[str],
        db=None,
        model=None,
        which: str = "M1",
        *,
        show_progress: bool = False,
    ) -> "pd.DataFrame":
        """
        Batch-classify a list of variant IDs. Returns a DataFrame.

        Handles both single-variant and epistasis IDs.
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        iterator: Iterable = mut_ids
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(mut_ids, desc="Classifying")
            except ImportError:
                pass

        for mid in iterator:
            delta = _get_delta_auto(
                mid, db=db, model=model, which=which,
                context=self.context, genome=self.genome, pool=self.pool,
            )
            if delta is None:
                logger.warning("Skipping %s (could not load/compute)", mid)
                continue
            result = self.classify(delta)
            row = {"mut_id": mid, **result.to_dict()}
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    def validate(self, cv: int = 0, seed: int = 42) -> DeltaValidationResult:
        """
        Evaluate the classifier.

        If ``cv > 0``, performs stratified k-fold cross-validation
        (retraining from scratch each fold, including re-fitting
        preprocessing) and reports fold accuracies.

        Always reports training-set accuracy and confusion matrix as
        a baseline.
        """
        import torch
        if self._X is None:
            raise RuntimeError("No data to validate on")

        X_raw, y = self._X.copy(), self._y.copy()

        # Apply the fitted preprocessor for full-data evaluation
        X = self.preprocessor.transform(X_raw)

        # Full-data accuracy (train set)
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.net(xt)
            preds = logits.argmax(1).cpu().numpy()

        overall_acc = float((preds == y).mean())
        per_class = {}
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for i in range(self.num_classes):
            mask = y == i
            if mask.sum() > 0:
                per_class[self.labels[i]] = float((preds[mask] == i).mean())
            else:
                per_class[self.labels[i]] = 0.0
            for j in range(self.num_classes):
                cm[i, j] = int(((y == i) & (preds == j)).sum())

        # Cross-validation
        cv_accuracy = None
        cv_std = None
        cv_per_fold = None

        if cv > 0:
            try:
                from sklearn.model_selection import StratifiedKFold
            except ImportError:
                warnings.warn("sklearn not installed; skipping CV")
                cv = 0

        if cv > 0:
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
            fold_accs = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
                # Fit preprocessing on THIS fold's training data
                fold_pp = _Preprocessor(
                    mode=self.preprocessor.mode,
                    whiten_k=self.preprocessor.whiten_k,
                    whiten_variance=self.preprocessor.whiten_variance,
                )
                fold_pp.fit(X_raw[train_idx])

                fold_X_train = fold_pp.transform(X_raw[train_idx])
                fold_X_test = fold_pp.transform(X_raw[test_idx])

                eff_dim = fold_pp.output_dim_

                # Build a fresh MLP for this fold
                fold_net = _build_mlp(
                    eff_dim, self.num_classes,
                    hidden_dims=self._hidden_dims,
                    dropout=self._dropout,
                    batch_norm=self._batch_norm,
                ).to(self.device)

                fold_X_train_t = torch.tensor(fold_X_train, dtype=torch.float32)
                fold_y_train_t = torch.tensor(y[train_idx], dtype=torch.long)
                fold_X_test_t = torch.tensor(fold_X_test, dtype=torch.float32)

                # Class weights
                counts = np.bincount(y[train_idx], minlength=self.num_classes).astype(np.float64)
                counts = np.maximum(counts, 1)
                w = (1.0 / counts) * len(y[train_idx]) / self.num_classes
                weight_tensor = torch.tensor(w, dtype=torch.float32).to(self.device)

                criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
                optimizer = torch.optim.AdamW(fold_net.parameters(), lr=1e-3, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

                ds = torch.utils.data.TensorDataset(fold_X_train_t, fold_y_train_t)
                loader = torch.utils.data.DataLoader(
                    ds, batch_size=256, shuffle=True,
                    generator=torch.Generator().manual_seed(seed + fold_idx),
                )

                # Train for a fixed number of epochs per fold
                fold_net.train()
                for _ in range(100):
                    for xb, yb in loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        loss = criterion(fold_net(xb), yb)
                        loss.backward()
                        optimizer.step()
                    sched.step()

                # Evaluate fold
                fold_net.eval()
                with torch.no_grad():
                    logits_test = fold_net(fold_X_test_t.to(self.device))
                    fold_preds = logits_test.argmax(1).cpu().numpy()
                    fold_acc = float((fold_preds == y[test_idx]).mean())
                fold_accs.append(fold_acc)
                logger.info("Fold %d/%d accuracy: %.4f", fold_idx + 1, cv, fold_acc)

            cv_per_fold = fold_accs
            cv_accuracy = float(np.mean(fold_accs))
            cv_std = float(np.std(fold_accs))
            logger.info("CV accuracy: %.4f \u00b1 %.4f", cv_accuracy, cv_std)

        return DeltaValidationResult(
            accuracy=overall_acc,
            per_class_accuracy=per_class,
            confusion_matrix=cm,
            labels=list(self.labels),
            cv_accuracy=cv_accuracy,
            cv_std=cv_std,
            cv_per_fold=cv_per_fold,
        )

    # ------------------------------------------------------------------ #
    #  Plotting
    # ------------------------------------------------------------------ #

    def plot(
        self,
        method: str = "pca",
        figsize: Tuple[int, int] = (8, 6),
        title: Optional[str] = None,
        ax=None,
    ):
        """
        2D scatter of the training deltas coloured by class.

        Parameters
        ----------
        method : "pca" or "umap"
        """
        if self._X is None:
            raise RuntimeError("No data to plot")

        import matplotlib.pyplot as plt

        # Plot in the preprocessed space
        X = self.preprocessor.transform(self._X.copy())

        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(X)
                xlabel, ylabel = "UMAP-1", "UMAP-2"
            except ImportError:
                warnings.warn("umap not installed, falling back to PCA")
                method = "pca"

        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for i, label in enumerate(self.labels):
            mask = self._y == i
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                label=f"{label} (n={mask.sum()})",
                alpha=0.5, s=10,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, markerscale=2)
        ax.set_title(title or "Delta embedding space")
        plt.tight_layout()
        return ax

    def plot_training(
        self,
        result: TrainResult,
        figsize: Tuple[int, int] = (12, 4),
    ):
        """Plot training and validation loss/accuracy curves."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        epochs = range(1, len(result.train_losses) + 1)
        ax1.plot(epochs, result.train_losses, label="Train")
        ax1.plot(epochs, result.val_losses, label="Val")
        ax1.axvline(result.best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_title("Loss")

        ax2.plot(epochs, result.train_accs, label="Train")
        ax2.plot(epochs, result.val_accs, label="Val")
        ax2.axvline(result.best_epoch, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.set_title("Accuracy")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  Save / Load
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the classifier (architecture + weights + preprocessing + metadata).
        """
        import torch
        path = Path(path)
        state = {
            "raw_input_dim": self.raw_input_dim,
            "effective_input_dim": self._effective_input_dim,
            "labels": self.labels,
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
            "batch_norm": self._batch_norm,
            "preprocessor": self.preprocessor.state_dict(),
            "context": self.context,
            "genome": self.genome,
            "pool": self.pool,
            "net_state_dict": self.net.state_dict(),
            "fitted": self._fitted,
        }
        # Optionally store training data for later CV
        if self._X is not None:
            state["X"] = self._X
            state["y"] = self._y
            state["group_ids"] = self._group_ids
        torch.save(state, path)
        logger.info("Saved DeltaClassifier to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "DeltaClassifier":
        """
        Load a saved classifier.
        """
        import torch
        path = Path(path)
        state = torch.load(path, map_location=device, weights_only=False)

        pp_state = state.get("preprocessor", {})
        pp = _Preprocessor.from_state_dict(pp_state) if pp_state else _Preprocessor("none")

        obj = cls(
            input_dim=state.get("raw_input_dim", state.get("input_dim", 0)),
            labels=state["labels"],
            hidden_dims=state["hidden_dims"],
            dropout=state["dropout"],
            batch_norm=state["batch_norm"],
            preprocessing=pp.mode,
            context=state.get("context", _DEFAULT_CONTEXT),
            genome=state.get("genome", _DEFAULT_GENOME),
            pool=state.get("pool", "mean"),
        )
        obj.preprocessor = pp
        eff_dim = state.get("effective_input_dim", pp.output_dim_ or obj.raw_input_dim)
        obj._rebuild_net(eff_dim)
        obj.net.load_state_dict(state["net_state_dict"])
        obj.net = obj.net.to(device)
        obj.device = torch.device(device)
        obj._fitted = state.get("fitted", True)
        if "X" in state:
            obj._X = state["X"]
            obj._y = state["y"]
            obj._group_ids = state.get("group_ids")
        obj.net.eval()
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        n = len(self._X) if self._X is not None else 0
        pp = self.preprocessor.mode
        return (
            f"DeltaClassifier({status}, classes={self.labels}, "
            f"n_train={n}, raw_dim={self.raw_input_dim}, "
            f"effective_dim={self._effective_input_dim}, "
            f"preprocessing={pp!r})"
        )
