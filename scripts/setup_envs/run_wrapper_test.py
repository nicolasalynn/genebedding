#!/usr/bin/env python3
"""
Run a single wrapper test: import wrapper, instantiate, embed a short test sequence.
Prints one line: WRAPPER_KEY OK shape=(...)  or  WRAPPER_KEY FAIL <message>
Exit code 0 on success, 1 on failure. Used by test_all_envs.sh.

With --report-params, outputs enriched tab-separated data to stdout:
  key<TAB>STATUS<TAB>emb_dim<TAB>params<TAB>context_bp
"""
from __future__ import annotations

import argparse
import sys

# Test sequence (short; wrappers that need longer input pad or crop)
DEFAULT_SEQ = "ACGT" * 100   # 400 bp

# Hardcoded context lengths from FULL_MODEL_CONFIG
CONTEXT_BP = {
    "convnova": 1000, "mutbert": 256, "hyenadna": 60000, "caduceus": 60000,
    "dnabert": 512, "rinalmo": 511, "specieslm": 600, "borzoi": 260000,
    "alphagenome": 4096, "evo2": 4000, "spliceai": 10000,
    # NT variants all use 3000bp context
    "nt50_3mer": 3000, "nt50_multi": 3000, "nt100_multi": 3000,
    "nt250_multi": 3000, "nt500_multi": 3000, "nt500_ref": 3000,
    "nt2500_multi": 3000, "nt2500_okgp": 3000,
}


def _count_params(wrapper_key: str, w) -> int | None:
    """Count total parameters for a wrapper's model."""
    try:
        if wrapper_key == "nt":
            return sum(p.numel() for p in w.mlm.parameters())
        elif wrapper_key == "spliceai":
            total = 0
            for m in w._models:
                total += sum(p.numel() for p in m.parameters())
            return total
        elif wrapper_key == "alphagenome":
            import jax
            leaves = jax.tree_util.tree_leaves(w._params)
            return sum(x.size for x in leaves)
        else:
            return sum(p.numel() for p in w.model.parameters())
    except Exception:
        return None


def run_test(wrapper_key: str, model: str | None = None, seq: str = DEFAULT_SEQ,
             report_params: bool = False) -> bool:
    # Determine the report key (for display and CONTEXT_BP lookup)
    if wrapper_key == "nt":
        report_key = model or "nt2500_multi"
    else:
        report_key = wrapper_key

    try:
        if wrapper_key == "nt":
            import os
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token:
                try:
                    from huggingface_hub import login
                    login(token=token)
                except Exception:
                    pass
            from genebeddings.wrappers import NTWrapper
            nt_model = model or "nt2500_multi"
            w = NTWrapper(model=nt_model)
        elif wrapper_key == "alphagenome":
            from genebeddings.wrappers import AlphaGenomeWrapper
            w = AlphaGenomeWrapper(model_version="fold_0", source="huggingface")
            # AlphaGenome needs longer sequences (128bp resolution attention)
            seq = "ACGT" * 512  # 2048 bp
        elif wrapper_key == "evo2":
            from genebeddings.wrappers import Evo2Wrapper
            w = Evo2Wrapper(model="7b_base")
        elif wrapper_key == "spliceai":
            import os
            if not os.environ.get("OPENSPLICEAI_MODEL_DIR"):
                if report_params:
                    ctx = CONTEXT_BP.get(report_key, "N/A")
                    print(f"{report_key}\tSKIP\tN/A\tN/A\t{ctx}")
                else:
                    print(f"{report_key} SKIP OPENSPLICEAI_MODEL_DIR not set", file=sys.stderr)
                return True  # skip, do not fail the run
            from genebeddings.wrappers import SpliceAIWrapper
            w = SpliceAIWrapper()
        elif wrapper_key == "convnova":
            from genebeddings.wrappers import ConvNovaWrapper
            w = ConvNovaWrapper()
        elif wrapper_key == "borzoi":
            from genebeddings.wrappers import BorzoiWrapper
            w = BorzoiWrapper(repo="johahi/flashzoi-replicate-0")
        elif wrapper_key == "mutbert":
            from genebeddings.wrappers import MutBERTWrapper
            w = MutBERTWrapper(model="mutbert")
        elif wrapper_key == "hyenadna":
            from genebeddings.wrappers import HyenaDNAWrapper
            w = HyenaDNAWrapper(model="medium-160k")
        elif wrapper_key == "caduceus":
            from genebeddings.wrappers import CaduceusWrapper
            w = CaduceusWrapper()
        elif wrapper_key == "dnabert":
            from genebeddings.wrappers import DNABERTWrapper
            w = DNABERTWrapper()
        elif wrapper_key == "rinalmo":
            from genebeddings.wrappers import RiNALMoWrapper
            w = RiNALMoWrapper(model_name="giga-v1")
        elif wrapper_key == "specieslm":
            from genebeddings.wrappers import SpeciesLMWrapper
            w = SpeciesLMWrapper()
        else:
            if report_params:
                print(f"{report_key}\tFAIL\tN/A\tN/A\tN/A")
            print(f"{report_key} FAIL unknown wrapper key", file=sys.stderr)
            return False

        emb = w.embed(seq, pool="mean")
        shape = getattr(emb, "shape", None)
        emb_dim = shape[-1] if shape is not None else "N/A"

        if report_params:
            params = _count_params(wrapper_key, w)
            params_str = str(params) if params is not None else "N/A"
            ctx = CONTEXT_BP.get(report_key, "N/A")
            print(f"{report_key}\tOK\t{emb_dim}\t{params_str}\t{ctx}")
        else:
            print(f"{report_key} OK shape={shape}")
        return True
    except Exception as e:
        if report_params:
            ctx = CONTEXT_BP.get(report_key, "N/A")
            print(f"{report_key}\tFAIL\tN/A\tN/A\t{ctx}")
        print(f"{report_key} FAIL {e!s}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Run one wrapper test for test_all_envs.sh")
    ap.add_argument("--wrapper", required=True,
                    help="Wrapper key: nt, alphagenome, evo2, spliceai, convnova, borzoi, "
                         "mutbert, hyenadna, caduceus, dnabert, rinalmo, specieslm")
    ap.add_argument("--model", default=None,
                    help="Model variant (e.g. nt50_3mer for NT wrapper). "
                         "If omitted, uses wrapper default.")
    ap.add_argument("--seq-len", type=int, default=100,
                    help="Test sequence length in units of 'ACGT' (default 100 -> 400 bp)")
    ap.add_argument("--report-params", action="store_true",
                    help="Output enriched TSV: key<TAB>STATUS<TAB>emb_dim<TAB>params<TAB>context_bp")
    args = ap.parse_args()
    seq = "ACGT" * max(1, args.seq_len)
    ok = run_test(args.wrapper, model=args.model, seq=seq, report_params=args.report_params)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
