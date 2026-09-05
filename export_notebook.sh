#!/bin/sh
# Export a notebook to HTML with ONLY its output cells (figures, tables,
# printouts) -- all source code stripped via nbconvert's --no-input.
#
# Usage:
#   ./export_notebook.sh <path/to/notebook.ipynb | path/to/notebook.py> [output.html]
#
# Accepts either an executed .ipynb OR a jupytext percent .py file:
#   - .ipynb : uses the notebook's already-stored outputs (does NOT re-execute);
#              run the notebook first if you want fresh outputs.
#   - .py    : a jupytext (formats: ipynb,py:percent) source. It is converted to
#              a paired .ipynb AND executed (so outputs exist) before export. The
#              generated .ipynb is left next to the .py.
#
# If [output.html] is omitted, writes alongside the input as <name>.outputs.html.
# The output path is relative to your current directory (not the input's), and
# parent dirs are created.

set -e

in="$1"
if [ -z "$in" ]; then
    echo "usage: $0 <path/to/notebook.ipynb | path/to/notebook.py> [output.html]" >&2
    exit 1
fi
if [ ! -f "$in" ]; then
    echo "error: input not found: $in" >&2
    exit 1
fi

# Resolve the input to an executed .ipynb (`nb`). A jupytext .py is converted and
# executed to a paired notebook; an .ipynb is used as-is (stored outputs).
case "$in" in
    *.py)
        nb="${in%.py}.ipynb"
        echo "jupytext: converting + executing $in -> $nb"
        uv run --with matplotlib --with pandas --with jupyter \
               --with nbconvert --with jupytext --with ipykernel \
            jupytext --to notebook --execute --output "$nb" "$in"
        ;;
    *.ipynb)
        nb="$in"
        ;;
    *)
        echo "error: expected a .ipynb or .py (jupytext) file: $in" >&2
        exit 1
        ;;
esac

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
