#!/usr/bin/env bash
# build.sh — render paper/manuscript.md to PDF (and to manuscript.tex).
#
# Usage:
#   bash paper/build.sh         # produces paper/manuscript.pdf and paper/manuscript.tex
#   bash paper/build.sh tex     # produces paper/manuscript.tex only (no PDF compile)
#
# Approach:
#   - References are hand-formatted as a bibliography section in manuscript.md,
#     so we do not rely on pandoc-citeproc (Pandoc 2.9 lacks built-in citeproc).
#   - For Biological Cybernetics submission, the resulting manuscript.tex
#     should be embedded in Springer Nature's `sn-jnl.cls` template separately.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

MD="manuscript.md"
TEX="manuscript.tex"
PDF="manuscript.pdf"

# Use XeLaTeX so Unicode (λ, γ, μ, ±, …) renders natively.
# preamble.tex forces \boldsymbol to use unicode-math's \symbf, otherwise
# Greek vector symbols (\boldsymbol{\lambda}, \boldsymbol{\tau}, ...) fall
# back to regular italic and are visually indistinguishable from scalars.
PANDOC_OPTS=(
  --top-level-division=section
  --standalone
  --include-in-header=preamble.tex
  -V geometry:a4paper,margin=25mm
  -V fontsize=11pt
  -V documentclass=article
  -V mainfont="DejaVu Serif"
  -V mathfont="Latin Modern Math"
  -V linestretch=1.15
  -V colorlinks=true
)

if [[ "${1:-pdf}" == "tex" ]]; then
  pandoc "$MD" "${PANDOC_OPTS[@]}" -o "$TEX"
  echo "→ $TEX"
  exit 0
fi

# Build PDF directly via pandoc → xelatex (Unicode-safe).
pandoc "$MD" "${PANDOC_OPTS[@]}" --pdf-engine=xelatex -o "$PDF"
echo "→ $PDF"

# Also keep the .tex artefact for downstream Springer template work.
pandoc "$MD" "${PANDOC_OPTS[@]}" -o "$TEX"
echo "→ $TEX"
