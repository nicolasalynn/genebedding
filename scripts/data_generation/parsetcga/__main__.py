"""Run with: python -m parsetcga  → prebuilds the mutation lookup DB."""
from .mutations import build_mutation_lookup

if __name__ == "__main__":
    print("Building mutation lookup (one-time, may take a few minutes)...")
    path = build_mutation_lookup()
    print("Done. Lookup DB:", path)
