#!/bin/sh
# Export a notebook to HTML with ONLY its output cells (figures, tables,
# printouts) -- all source code stripped via nbconvert's --no-input.
#
# Usage:
#   ./export_outputs.sh <path/to/notebook.ipynb> [output.html]
#
# Uses the notebook's already-stored outputs (does NOT re-execute). Run the
# notebook first if you want fresh outputs. If [output.html] is omitted, writes
# alongside the notebook as <name>.outputs.html. The output path is relative to
# your current directory (not the notebook's), and parent dirs are created.

set -e

nb="$1"
if [ -z "$nb" ]; then
    echo "usage: $0 <path/to/notebook.ipynb> [output.html]" >&2
    exit 1
fi
if [ ! -f "$nb" ]; then
    echo "error: notebook not found: $nb" >&2
    exit 1
fi

out="$2"
if [ -z "$out" ]; then
    out="${nb%.ipynb}.outputs.html"
fi

# Split into directory + basename so we can point nbconvert's --output-dir at a
# path relative to the CWD (its --output alone is resolved relative to the
# notebook's own directory, which is surprising). Strip a trailing .html since
# the html exporter appends it.
outdir=$(dirname "$out")
base=$(basename "$out")
base=${base%.html}

mkdir -p "$outdir"

uv run --with matplotlib --with pandas --with jupyter --with nbconvert \
    jupyter nbconvert --to html --no-input \
    --output-dir "$outdir" --output "$base" "$nb"

echo "wrote $outdir/$base.html"
