# alphagenome_wrapper.py
# AlphaGenome wrapper for genomic foundation models
# Standardized API: embed(), predict_tracks()
#
# AlphaGenome is a JAX-based DNA sequence model from Google DeepMind that
# predicts regulatory variant effects at single base-pair resolution across
# gene expression, splicing, chromatin features, and contact maps.
# Sequences up to 1M bp. Requires GPU (NVIDIA H100 recommended).

import re
from typing import Union, Literal

import numpy as np

# Support both package import and direct sys.path import
try:
    from .base_wrapper import BaseWrapper, PoolMode
except ImportError:
    from base_wrapper import BaseWrapper, PoolMode


# Model versions available from Kaggle / HuggingFace
ALPHAGENOME_VERSIONS = [
    "all_folds",
    "fold_0",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
]

AlphaGenomeSource = Literal["kaggle", "huggingface"]


class AlphaGenomeWrapper(BaseWrapper):
    """
    AlphaGenome wrapper with standardized API.

    AlphaGenome is a unified DNA sequence foundation model from Google DeepMind
    that analyzes sequences up to 1M bp at single base-pair resolution. It uses
    a JAX-based encoder-transformer-decoder architecture producing embeddings at
    1bp resolution (1536-dim) and 128bp resolution (3072-dim), plus pairwise
    embeddings (128-dim).

    Implements BaseWrapper: embed(), predict_tracks()

    Parameters
    ----------
    model_version : str, default="all_folds"
        Model version/fold to load. One of: "all_folds", "fold_0", ..., "fold_4".
    source : {'kaggle', 'huggingface'}, default='huggingface'
        Where to download model weights from.
    organism_settings : dict, optional
        Custom organism settings. If None, uses AlphaGenome defaults
        (human and mouse reference genomes).
    device : jax.Device, optional
        JAX device for inference. If None, uses first available GPU/TPU.
        Note: AlphaGenome strongly recommends GPU (H100+) or TPU.

    Examples
    --------
    >>> wrapper = AlphaGenomeWrapper(model_version="all_folds")
    >>> emb = wrapper.embed("ACGT" * 256, pool="mean")  # (1536,) numpy array
    >>> emb_128 = wrapper.embed("ACGT" * 256, pool="mean", resolution=128)  # (3072,)
    """

    def __init__(
        self,
        model_version: str = "all_folds",
        *,
        source: AlphaGenomeSource = "huggingface",
        organism_settings=None,
        device=None,
        fixed_length: int | None = None,
        pad_to_multiple: int | None = None,
        pad_char: str = "N",
    ):
        super().__init__()

        import jax
        import jmp
        import haiku as hk
        from alphagenome_research.model import dna_model
        from alphagenome_research.model import model as model_module
        from alphagenome_research.model import one_hot_encoder

        self.model_version = model_version
        self.fixed_length = fixed_length
        self.pad_to_multiple = pad_to_multiple
        self.pad_char = pad_char
        self._one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
        self._dna_model = dna_model

        kwargs = {}
        if organism_settings is not None:
            kwargs["organism_settings"] = organism_settings
        if device is not None:
            kwargs["device"] = device

        if source == "kaggle":
            self._model = dna_model.create_from_kaggle(
                model_version, **kwargs
            )
        elif source == "huggingface":
            self._model = dna_model.create_from_huggingface(
                model_version, **kwargs
            )
        else:
            raise ValueError(f"source must be 'kaggle' or 'huggingface', got {source!r}")

        # Store references for direct JAX access
        self._params = self._model._params
        self._state = self._model._state
        self._device_context = self._model._device_context
        self._metadata = self._model._metadata

        # Build a dedicated embedding extraction function.
        # The standard _apply_fn discards the Embeddings object returned by
        # AlphaGenome.__call__. We build our own that returns just embeddings.
        jmp_policy = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
        metadata = self._metadata

        @hk.transform_with_state
        def _forward_embed(dna_sequence, organism_index):
            with hk.mixed_precision.push_policy(model_module.AlphaGenome, jmp_policy):
                _, embeddings = model_module.AlphaGenome(metadata)(
                    dna_sequence, organism_index
                )
            return embeddings

        def _embed_apply_fn(params, state, dna_sequence, organism_index):
            (embeddings, _), _ = _forward_embed.apply(
                params, state, None, dna_sequence, organism_index
            )
            return embeddings

        self._embed_fn = jax.jit(_embed_apply_fn)

    def __repr__(self) -> str:
        return (
            f"AlphaGenomeWrapper(model_version='{self.model_version}', "
            f"capabilities={self.get_capabilities()})"
        )

    @staticmethod
    def _normalize_seq(seq: str) -> str:
        """Clean sequence to valid DNA characters (uppercase ACGT, others become N)."""
        return re.sub(r"[^ACGTacgt]", "N", seq or "").upper()

    def _pad_or_crop(self, seq: str, *, fixed_length: int | None, pad_to_multiple: int | None) -> str:
        """Pad/crop sequence to a fixed length or to a multiple."""
        if fixed_length is not None:
            L = len(seq)
            if L == fixed_length:
                return seq
            if L > fixed_length:
                # Center-crop to preserve variant-centered windows
                start = max(0, (L - fixed_length) // 2)
                return seq[start:start + fixed_length]
            # Center-pad
            pad_total = fixed_length - L
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return (self.pad_char * pad_left) + seq + (self.pad_char * pad_right)

        if pad_to_multiple:
            L = len(seq)
            remainder = L % pad_to_multiple
            if remainder == 0:
                return seq
            pad_total = pad_to_multiple - remainder
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return (self.pad_char * pad_left) + seq + (self.pad_char * pad_right)

        return seq

    def _encode_sequence(self, seq: str) -> np.ndarray:
        """One-hot encode a DNA sequence to shape (S, 4) float32."""
        seq = self._normalize_seq(seq)
        seq = self._pad_or_crop(
            seq,
            fixed_length=self.fixed_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return np.asarray(self._one_hot_encoder.encode(seq), dtype=np.float32)

    def embed(
        self,
        seq: str,
        *,
        pool: PoolMode = "mean",
        return_numpy: bool = True,
        resolution: int = 1,
        organism: str = "HOMO_SAPIENS",
        fixed_length: int | None = None,
        pad_to_multiple: int | None = None,
    ) -> Union[np.ndarray, "jax.Array"]:
        """
        Generate embeddings for a DNA sequence.

        AlphaGenome produces embeddings at two resolutions:
        - 1bp: shape (S, 1536) — from the decoder
        - 128bp: shape (S//128, 3072) — from the transformer trunk

        Parameters
        ----------
        seq : str
            Input DNA sequence (up to ~1M bp).
        pool : {'mean', 'cls', 'tokens'}, default='mean'
            Pooling strategy:
            - 'mean': Average over all positions (returns 1D vector)
            - 'cls': Use first position embedding (returns 1D vector)
            - 'tokens': Return all position embeddings (returns 2D array)
        return_numpy : bool, default=True
            If True, return numpy array. (JAX arrays returned otherwise.)
        resolution : {1, 128}, default=1
            Embedding resolution in base pairs.
            - 1: 1bp resolution, 1536-dim embeddings
            - 128: 128bp resolution, 3072-dim embeddings
        organism : str, default='HOMO_SAPIENS'
            Organism for organism-specific embeddings.
            One of 'HOMO_SAPIENS', 'MUS_MUSCULUS'.
        fixed_length : int, optional
            If provided, pad/crop sequence to this length before embedding.
            Overrides the wrapper default.
        pad_to_multiple : int, optional
            If provided, pad sequence length up to a multiple of this value.
            Overrides the wrapper default.

        Returns
        -------
        embeddings : np.ndarray or jax.Array
            Shape depends on pool and resolution:
            - 'mean' or 'cls': (hidden_dim,) where hidden_dim is 1536 (1bp) or 3072 (128bp)
            - 'tokens': (num_positions, hidden_dim)
        """
        import jax

        if resolution not in (1, 128):
            raise ValueError(f"resolution must be 1 or 128, got {resolution}")
        if pool not in ("mean", "cls", "tokens"):
            raise ValueError(f"pool must be 'mean', 'cls', or 'tokens', got {pool!r}")

        organism_enum = self._dna_model.Organism[organism]
        organism_idx = self._dna_model.convert_to_organism_index(organism_enum)

        if fixed_length is not None or pad_to_multiple is not None:
            seq = self._normalize_seq(seq)
            seq = self._pad_or_crop(seq, fixed_length=fixed_length, pad_to_multiple=pad_to_multiple)
            encoded = np.asarray(self._one_hot_encoder.encode(seq), dtype=np.float32)
        else:
            encoded = self._encode_sequence(seq)

        with self._device_context as device, jax.transfer_guard("disallow"):
            sequence = jax.device_put(encoded[np.newaxis], device)
            organism_index = jax.device_put(
                np.full((1,), organism_idx, dtype=np.int32), device
            )

            embeddings = self._embed_fn(
                self._params, self._state, sequence, organism_index
            )

        # Select resolution
        emb = embeddings.get_sequence_embeddings(resolution)  # (1, L, D)
        emb = emb[0]  # (L, D) — remove batch dimension

        # Upcast from bfloat16 to float32
        emb = emb.astype(np.float32) if return_numpy else emb

        if pool == "tokens":
            result = emb
        elif pool == "cls":
            result = emb[0]
        elif pool == "mean":
            result = emb.mean(axis=0)

        if return_numpy:
            return np.asarray(result, dtype=np.float32)
        return result

    def predict_tracks(
        self,
        seq: str,
        *,
        organism: str = "HOMO_SAPIENS",
        requested_outputs=None,
        ontology_terms=None,
    ):
        """
        Predict genomic tracks for a DNA sequence.

        Returns the full AlphaGenome Output object containing predictions across
        all requested modalities (RNA-seq, ATAC, DNase, ChIP, splice sites, etc.).

        Parameters
        ----------
        seq : str
            Input DNA sequence (up to ~1M bp).
        organism : str, default='HOMO_SAPIENS'
            Organism for predictions. One of 'HOMO_SAPIENS', 'MUS_MUSCULUS'.
        requested_outputs : list of OutputType, optional
            Which output types to predict. If None, predicts all available.
        ontology_terms : list of str, optional
            Ontology terms (e.g., 'UBERON:0001157') to filter tracks.

        Returns
        -------
        output : dna_output.Output
            AlphaGenome Output object with track predictions.
        """
        from alphagenome.models import dna_output

        organism_enum = self._dna_model.Organism[organism]

        if requested_outputs is None:
            requested_outputs = list(dna_output.OutputType)

        return self._model.predict_sequence(
            seq,
            organism=organism_enum,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

    def predict_variant(
        self,
        interval,
        variant,
        *,
        organism: str = "HOMO_SAPIENS",
        requested_outputs=None,
        ontology_terms=None,
    ):
        """
        Predict variant effects by comparing reference and alternate sequences.

        Parameters
        ----------
        interval : genome.Interval
            Genomic interval (chromosome, start, end).
        variant : genome.Variant
            Genetic variant (chromosome, position, ref, alt).
        organism : str, default='HOMO_SAPIENS'
            Organism for predictions.
        requested_outputs : list of OutputType, optional
            Which output types to predict. If None, predicts all available.
        ontology_terms : list of str, optional
            Ontology terms to filter tracks.

        Returns
        -------
        output : dna_output.VariantOutput
            Output with reference and alternate predictions.
        """
        from alphagenome.models import dna_output

        organism_enum = self._dna_model.Organism[organism]

        if requested_outputs is None:
            requested_outputs = list(dna_output.OutputType)

        return self._model.predict_variant(
            interval,
            variant,
            organism=organism_enum,
            requested_outputs=requested_outputs,
            ontology_terms=ontology_terms,
        )

    def score_variant(
        self,
        interval,
        variant,
        variant_scorers=(),
        *,
        organism: str = "HOMO_SAPIENS",
    ) -> list:
        """
        Score variant effects using AlphaGenome's built-in variant scorers.

        Parameters
        ----------
        interval : genome.Interval
            Genomic interval.
        variant : genome.Variant
            Genetic variant.
        variant_scorers : sequence of VariantScorerTypes, optional
            Scorers to use. If empty, uses recommended scorers.
        organism : str, default='HOMO_SAPIENS'
            Organism for predictions.

        Returns
        -------
        scores : list of anndata.AnnData
            Variant effect scores from each scorer.
        """
        organism_enum = self._dna_model.Organism[organism]
        return self._model.score_variant(
            interval,
            variant,
            variant_scorers,
            organism=organism_enum,
        )
