#!/usr/bin/env python3
"""
Run a single wrapper test: import wrapper, instantiate, embed a short test sequence.
Prints one line: WRAPPER_KEY OK shape=(...)  or  WRAPPER_KEY FAIL <message>
Exit code 0 on success, 1 on failure. Used by test_all_envs.sh.
"""
from __future__ import annotations

import argparse
import sys

# Test sequence (short; wrappers that need longer input pad or crop)
DEFAULT_SEQ = "ACGT" * 100   # 400 bp


def run_test(wrapper_key: str, seq: str = DEFAULT_SEQ) -> bool:
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
            # Use 2.5b multi-species (InstaDeepAI/nucleotide-transformer-2.5b-multi-species); gated, needs HF token
            w = NTWrapper(model="nt2500_multi")
        elif wrapper_key == "alphagenome":
            from genebeddings.wrappers import AlphaGenomeWrapper
            w = AlphaGenomeWrapper(model_version="fold_0", source="huggingface")
        elif wrapper_key == "evo2":
            from genebeddings.wrappers import Evo2Wrapper
            w = Evo2Wrapper(model="7b_base")
        elif wrapper_key == "spliceai":
            import os
            if not os.environ.get("OPENSPLICEAI_MODEL_DIR"):
                print(f"{wrapper_key} SKIP OPENSPLICEAI_MODEL_DIR not set", file=sys.stderr)
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
            print(f"{wrapper_key} FAIL unknown wrapper key", file=sys.stderr)
            return False

        emb = w.embed(seq, pool="mean")
        shape = getattr(emb, "shape", None) or "(no shape)"
        print(f"{wrapper_key} OK shape={shape}")
        return True
    except Exception as e:
        print(f"{wrapper_key} FAIL {e!s}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Run one wrapper test for test_all_envs.sh")
    ap.add_argument("--wrapper", required=True, help="Wrapper key: nt, alphagenome, evo2, spliceai, convnova, borzoi, mutbert, hyenadna, caduceus, dnabert, rinalmo, specieslm")
    ap.add_argument("--seq-len", type=int, default=100, help="Test sequence length in units of 'ACGT' (default 100 -> 400 bp)")
    args = ap.parse_args()
    seq = "ACGT" * max(1, args.seq_len)
    ok = run_test(args.wrapper, seq=seq)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
