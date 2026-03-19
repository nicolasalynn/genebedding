#!/usr/bin/env bash
# Download reference genome FASTAs for multi-genome distillation.
# Sources: UCSC, Ensembl, NCBI
#
# Usage:
#   bash scripts/distillation/download_genomes.sh /path/to/genomes
#   bash scripts/distillation/download_genomes.sh  # defaults to ~/data/genomes
#
# Downloads ~15 genomes spanning mammals, fish, insects, plants, fungi, bacteria.
# Total size: ~30GB compressed, ~80GB uncompressed.
# Add --small for a minimal set (5 genomes, ~10GB).

set -e

GENOME_DIR="${1:-$HOME/data/genomes}"
mkdir -p "$GENOME_DIR"
cd "$GENOME_DIR"

SMALL_MODE=false
if [[ "$2" == "--small" ]] || [[ "$1" == "--small" ]]; then
    SMALL_MODE=true
    GENOME_DIR="${2:-$HOME/data/genomes}"
    mkdir -p "$GENOME_DIR"
    cd "$GENOME_DIR"
fi

download() {
    local name="$1"
    local url="$2"
    local outfile="$3"

    if [ -f "$outfile" ]; then
        echo "  [skip] $name — already exists: $outfile"
        return
    fi

    echo "  [download] $name"
    if [[ "$url" == *.gz ]]; then
        wget -q "$url" -O "${outfile}.gz" && gunzip -f "${outfile}.gz"
    else
        wget -q "$url" -O "$outfile"
    fi
    echo "  [done] $outfile ($(du -h "$outfile" | cut -f1))"
}

echo "=== Downloading reference genomes to $GENOME_DIR ==="
echo ""

# ---- CORE SET (always downloaded) ----
echo "--- Core genomes ---"

# Human (likely already have this)
download "Human (hg38)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz" \
    "hg38.fa"

# Mouse
download "Mouse (mm39)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/mm39/bigZips/mm39.fa.gz" \
    "mm39.fa"

# Drosophila
download "Drosophila (dm6)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz" \
    "dm6.fa"

# Zebrafish
download "Zebrafish (danRer11)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/danRer11/bigZips/danRer11.fa.gz" \
    "danRer11.fa"

# C. elegans
download "C. elegans (ce11)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/ce11/bigZips/ce11.fa.gz" \
    "ce11.fa"

if [ "$SMALL_MODE" = true ]; then
    echo ""
    echo "=== Small mode: 5 genomes downloaded ==="
    ls -lh "$GENOME_DIR"/*.fa
    echo ""
    echo "Use with distillation:"
    echo "  --fasta $GENOME_DIR/hg38.fa \\"
    echo "  --extra-fastas $GENOME_DIR/mm39.fa $GENOME_DIR/dm6.fa $GENOME_DIR/danRer11.fa $GENOME_DIR/ce11.fa"
    exit 0
fi

# ---- EXTENDED SET ----
echo ""
echo "--- Extended genomes ---"

# Chicken
download "Chicken (galGal6)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/galGal6/bigZips/galGal6.fa.gz" \
    "galGal6.fa"

# Rat
download "Rat (rn7)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/rn7/bigZips/rn7.fa.gz" \
    "rn7.fa"

# Dog
download "Dog (canFam6)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/canFam6/bigZips/canFam6.fa.gz" \
    "canFam6.fa"

# Arabidopsis (plant)
download "Arabidopsis (TAIR10)" \
    "https://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/arabidopsis_thaliana/dna/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz" \
    "arabidopsis_tair10.fa"

# Rice (plant)
download "Rice (IRGSP-1.0)" \
    "https://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/oryza_sativa/dna/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa.gz" \
    "rice_irgsp.fa"

# Yeast
download "Yeast (sacCer3)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/sacCer3/bigZips/sacCer3.fa.gz" \
    "sacCer3.fa"

# E. coli (bacteria)
download "E. coli (K-12 MG1655)" \
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz" \
    "ecoli_k12.fa"

# Gorilla
download "Gorilla (gorGor6)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/gorGor6/bigZips/gorGor6.fa.gz" \
    "gorGor6.fa"

# Macaque
download "Macaque (rheMac10)" \
    "https://hgdownload.soe.ucsc.edu/goldenPath/rheMac10/bigZips/rheMac10.fa.gz" \
    "rheMac10.fa"

# Corn (plant)
download "Corn/Maize (Zm-B73-v5)" \
    "https://ftp.ensemblgenomes.org/pub/plants/release-57/fasta/zea_mays/dna/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa.gz" \
    "maize_b73.fa"

# Index all FASTAs for pysam
echo ""
echo "--- Indexing FASTAs ---"
for fa in "$GENOME_DIR"/*.fa; do
    if [ ! -f "${fa}.fai" ]; then
        echo "  Indexing $fa"
        samtools faidx "$fa" 2>/dev/null || echo "  [warn] samtools not found, skip indexing $fa"
    fi
done

echo ""
echo "=== Done: $(ls "$GENOME_DIR"/*.fa | wc -l) genomes downloaded ==="
ls -lh "$GENOME_DIR"/*.fa
echo ""
echo "Use with distillation:"
EXTRAS=$(ls "$GENOME_DIR"/*.fa | grep -v hg38 | tr '\n' ' ')
echo "  --fasta $GENOME_DIR/hg38.fa \\"
echo "  --extra-fastas $EXTRAS"
