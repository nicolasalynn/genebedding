"""
Canonical mutation and epistasis IDs.

Mutation ID format: gene:chrom:pos:ref:alt
  - chrom does NOT contain 'chr' (e.g. "1", "X", "Y").
  - Optional strand suffix :P or :N (positive/negative) is accepted and ignored for canonical form.
Epistasis ID: join mutation_ids with '|', sorted for consistency (e.g. "TP53:17:123:G:A|KRAS:12:456:C:T").
"""
from __future__ import annotations

from typing import Union

# Optional strand suffix on mutation IDs; we normalize by stripping these
_STRAND_SUFFIXES = (":P", ":N")


def _strip_chr(chrom: str) -> str:
    """Return chromosome without leading 'chr' if present."""
    if chrom is None or not isinstance(chrom, str):
        return str(chrom) if chrom is not None else ""
    s = chrom.strip()
    if s.lower().startswith("chr"):
        return s[3:]
    return s


def mutation_id(
    gene: str,
    chrom: str,
    pos: Union[int, str],
    ref: str,
    alt: str,
) -> str:
    """
    Build canonical mutation ID: gene:chrom:pos:ref:alt.
    chrom is normalized to not contain 'chr' (e.g. chr1 -> 1).
    """
    ref_s = (str(ref).strip() if ref is not None else "").upper()
    alt_s = (str(alt).strip() if alt is not None else "").upper()
    return ":".join(
        [str(gene).strip(), _strip_chr(str(chrom)), str(pos), ref_s, alt_s]
    )


def normalize_mutation_id(mutation_id_str: str) -> str:
    """
    Return canonical mutation ID (gene:chrom:pos:ref:alt).
    If the ID has an optional strand suffix :P or :N, it is stripped and ignored.
    Ref and alt are uppercased to match the parquet/lookup convention.
    """
    s = str(mutation_id_str).strip()
    if not s:
        return s
    for suffix in _STRAND_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
            break
    parts = s.split(":", 4)
    if len(parts) == 5:
        gene, chrom, pos, ref, alt = parts
        return ":".join([gene, _strip_chr(chrom), pos, (ref or "").upper(), (alt or "").upper()])
    return s


def epistasis_id(mutation_ids: Union[list[str], set[str], tuple[str, ...]]) -> str:
    """
    Join mutation IDs into a single epistasis ID with '|'.
    Sorted for deterministic ordering. IDs with optional :P or :N suffix are normalized
    before deduping so e.g. "TP53:17:1:G:A:P" and "TP53:17:1:G:A" count as one.
    """
    canonical = sorted(set(normalize_mutation_id(m) for m in mutation_ids if m))
    return "|".join(canonical)


def parse_mutation_id(mutation_id_str: str) -> dict:
    """
    Parse a mutation ID into components.
    Accepts canonical form (gene:chrom:pos:ref:alt) or with optional strand suffix (mut_id:P or mut_id:N).
    Returns dict with keys: gene, chrom, pos, ref, alt; and strand if suffix was present ('P' or 'N').
    """
    s = str(mutation_id_str).strip()
    strand = None
    for suffix in _STRAND_SUFFIXES:
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
            strand = suffix[1]  # 'P' or 'N'
            break
    parts = s.split(":", 4)
    if len(parts) != 5:
        raise ValueError(f"Expected 5 colon-separated parts (or 5 + optional :P/:N), got {len(parts)}: {mutation_id_str!r}")
    gene, chrom, pos, ref, alt = parts
    out = {"gene": gene, "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt}
    if strand is not None:
        out["strand"] = strand
    return out
