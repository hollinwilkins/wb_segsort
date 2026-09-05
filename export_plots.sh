#!/bin/sh
# Export every plot/figure in a notebook as an individual PNG file.
#
# Usage:
#   ./export_plots.sh [--dpi N] <notebook.ipynb | notebook.py> [output_dir]
#
# Accepts either an executed .ipynb OR a jupytext percent .py file:
#   - .ipynb : extracts the notebook's already-stored image outputs (does NOT
#              re-execute), unless --dpi is given (see below).
#   - .py    : a jupytext (formats: ipynb,py:percent) source. Converted to a
#              paired .ipynb AND executed before extraction.
#
# --dpi N : re-render the figures at N dots-per-inch. Because this changes how
#           figures are rasterized, it forces execution: a .py is executed as
#           usual (its paired .ipynb gets the high-DPI outputs); an .ipynb is
#           executed into a throwaway copy so your original file is untouched.
#           Without --dpi, figures keep whatever DPI they were rendered at.
#
# PNGs are written as plot_01.png, plot_02.png, ... (in cell/output order). If
# [output_dir] is omitted, they go to <name>_plots/ next to the input.

set -e

dpi=""
in=""
outdir=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dpi)   dpi="$2"; shift 2 ;;
        --dpi=*) dpi="${1#--dpi=}"; shift ;;
        -*)      echo "error: unknown option: $1" >&2; exit 1 ;;
        *)
            if [ -z "$in" ]; then in="$1"; else outdir="$1"; fi
            shift ;;
    esac
done

if [ -z "$in" ]; then
    echo "usage: $0 [--dpi N] <notebook.ipynb | notebook.py> [output_dir]" >&2
    exit 1
fi
if [ ! -f "$in" ]; then
    echo "error: input not found: $in" >&2
    exit 1
fi
if [ -n "$dpi" ]; then
    case "$dpi" in
        ''|*[!0-9]*) echo "error: --dpi expects a positive integer, got '$dpi'" >&2; exit 1 ;;
    esac
    # matplotlib reads MATPLOTLIBRC on import; figure/savefig dpi drive the size
    # of inline PNGs produced during execution.
    rcdir=$(mktemp -d)
    printf 'figure.dpi: %s\nsavefig.dpi: %s\n' "$dpi" "$dpi" > "$rcdir/matplotlibrc"
    export MATPLOTLIBRC="$rcdir/matplotlibrc"
    echo "rendering figures at ${dpi} dpi"
fi

# Resolve the input to an executed .ipynb (`nb`).
case "$in" in
    *.py)
        nb="${in%.py}.ipynb"
        echo "jupytext: converting + executing $in -> $nb"
        uv run --with matplotlib --with pandas --with jupyter \
               --with nbconvert --with jupytext --with ipykernel \
            jupytext --to notebook --execute --output "$nb" "$in"
        ;;
    *.ipynb)
        if [ -n "$dpi" ]; then
            # re-execute a throwaway copy so the requested DPI takes effect
            # without mutating the user's notebook.
            tmpd=$(mktemp -d)
            nb="$tmpd/executed.ipynb"
            echo "nbconvert: executing $in at ${dpi} dpi"
            uv run --with matplotlib --with pandas --with jupyter \
                   --with nbconvert --with ipykernel \
                jupyter nbconvert --to notebook --execute \
                --output-dir "$tmpd" --output executed "$in"
        else
            nb="$in"
        fi
        ;;
    *)
        echo "error: expected a .ipynb or .py (jupytext) file: $in" >&2
        exit 1
        ;;
esac

if [ -z "$outdir" ]; then
    base=$(basename "$in")
    base=${base%.ipynb}
    base=${base%.py}
    outdir="$(dirname "$in")/${base}_plots"
fi

mkdir -p "$outdir"

# Walk every code cell's outputs and dump each embedded image/png to a file.
uv run --with nbformat python - "$nb" "$outdir" <<'PY'
import base64, sys
import nbformat

nb_path, out_dir = sys.argv[1], sys.argv[2]
nb = nbformat.read(nb_path, as_version=4)

count = 0
for cell in nb.cells:
    if cell.get("cell_type") != "code":
        continue
    for output in cell.get("outputs", []):
        png = output.get("data", {}).get("image/png")
        if not png:
            continue
        count += 1
        path = f"{out_dir}/plot_{count:02d}.png"
        with open(path, "wb") as f:
            f.write(base64.b64decode(png))
        print(f"  {path}")

print(f"wrote {count} PNG(s) to {out_dir}")
PY
