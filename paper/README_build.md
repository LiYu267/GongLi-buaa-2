# T20 Paper: Build Instructions

## File Structure

```
paper/
  main.tex             # Main LaTeX manuscript (ACML 2026 / JMLR format)
  references.bib        # Bibliography file
  jmlr.cls              # JMLR document class (from ACML template)
  acml26.bib            # ACML template sample bibliography (not used)
  figures/              # 8 figures in PNG format
    fig1_unosat_validation.png
    fig2_bayesian_posterior_map.png
    fig3_sensor_only_comparison.png
    fig4_kld_barplot.png
    fig5_jsd_conflict_map.png
    fig6_fusion_suppression_map.png
    fig7_consistency_map.png
    fig8_interval_comparison.png
  tables/               # (tables embedded in main.tex via booktabs)
  build/                # Output directory for compiled PDF
  README_build.md       # This file
```

## Prerequisites

- TeX Live 2024+ or MiKTeX with pdflatex
- Packages: jmlr, booktabs, longtable, amsmath, amssymb, graphicx, hyperref, natbib, geometry

## Compilation Steps

### Option A: Standard pdflatex + bibtex

```bash
cd paper/

# First pass
pdflatex main.tex

# Bibliography
bibtex main

# Two more passes for cross-references
pdflatex main.tex
pdflatex main.tex

# Move output
mv main.pdf build/final_T20_paper.pdf
```

### Option B: latexmk (recommended)

```bash
cd paper/
latexmk -pdf main.tex
mv main.pdf build/final_T20_paper.pdf
```

### Option C: Overleaf

1. Create a new project on overleaf.com
2. Upload all files from `paper/` (including jmlr.cls and the figures/ directory)
3. Set compiler to `pdfLaTeX`
4. Click "Recompile"

## Items Requiring Manual Attention

### 1. Author Information
Replace the placeholder in `main.tex` (lines ~25-30):
```latex
\author{
  \Name{Author Name} \Email{author@university.edu}\\
  \addr{Department of Statistics, University Name}
}
```

### 2. References
The `references.bib` file contains entries with verified DOIs for:
- Zhang et al. (2023), Remote Sensing
- Xiao et al. (2024), European Journal of Remote Sensing
- Dempster (1967), Annals of Mathematical Statistics
- Hoffman & Gelman (2014), JMLR
- Kullback & Leibler (1951), Annals of Mathematical Statistics

The following entries have approximate or unverified DOIs and should be double-checked:
- `roman2018` — verify the specific Román & Stokes reference for NASA Black Marble
- `lebedev2022` — verify the specific Mariupol S2 paper reference
- `gelman2006` — verify the HalfCauchy prior paper reference
- `lin1991` — verify JSD paper reference

### 3. Chinese Characters
If the title or any content contains Chinese characters, compile with `xelatex` instead of `pdflatex`:
```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```
You may need to add `\usepackage{xeCJK}` or `\usepackage{ctex}` to the preamble.

### 4. Figure Paths
Figures are referenced as `figures/figN_*.png` relative to the `paper/` directory.
Ensure the compiler's working directory is `paper/`.

### 5. Funding Acknowledgement
If applicable, uncomment and fill in the `\acks{}` command near the end of `main.tex`.

## Paper Statistics

- Target: ACML 2026 (JMLR Workshop and Conference Proceedings format)
- Language: English
- Length: approximately 8-10 pages with figures
- Sections: 7 main + appendix
- Figures: 8
- Tables: 5 (1 in main text + 4 in main/appendix)
