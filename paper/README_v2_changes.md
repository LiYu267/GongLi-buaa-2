# Paper v2 Changes from v1

## Compilation
- **v1**: No LaTeX compilation (LaTeX not installed). Pandoc-generated docx with broken math and images.
- **v2**: Full MiKTeX pdflatex + bibtex compilation. 15-page PDF with all references resolved, 0 errors, 0 undefined citations.

## Structural Changes

### Evidence Chain (v2 follows A→B→C1→C2→C3→D)
v1 had a standard Methods-then-Results structure but the narrative was linear and didn't emphasize the logical chain. v2 explicitly structures Results as a 5-step evidence chain:
- §4.1: S2 and VIIRS Provide Non-Equivalent Damage Signals (Fig 1)
- §4.2: UNOSAT Validation (Fig 2)
- §4.3: Bayesian Posterior Collapse (Fig 3, Table 2)
- §4.4: KLD/JSD Information Conflict (Fig 4, Table 3)
- §4.5: Bayesian vs D-S Behavior Under Conflict (Fig 5, Table 4)

### Introduction
- **v1**: Broad motivation, three contributions listed.
- **v2**: Sharper framing: "Multi-sensor fusion assumes sensors observe the same latent state — this fails when they measure different damage dimensions." Three research questions clearly numbered. Contributions tied to the three analytical modules.

### Results — Figure-Driven Rewrite
- **v1**: Each figure mentioned in 1-3 sentences.
- **v2**: Each figure discussed across 2-4 paragraphs, describing spatial patterns, key numerical values, and the specific conclusion it supports. Every subsection ends with an explicit takeaway sentence.

### Discussion
- **v1**: Four subsections that were somewhat generic.
- **v2**: Four subsections that follow the evidence chain: (5.1) Why conflict is not failure, (5.2) Why the single latent state assumption is too strong, (5.3) When multi-sensor fusion actually helps, (5.4) Future: multi-dimensional model. Figure 6 (conceptual model) placed in §5.4.

### Conclusion
- **v1**: Summary paragraph.
- **v2**: Restates the full evidence chain (A→B→C1→C2→C3→D) with each piece numbered and referenced to its supporting figure or table.

## Figure Changes

| Figure | v1 | v2 |
|--------|----|----|
| Fig 1 | UNOSAT ROC/PR only | **New composite**: S2 map + VIIRS map + scatter (3 panels, proves sensors disagree) |
| Fig 2 | Bayesian posterior map + sensor-only | **New composite**: ROC curves + AUC barplot (proves VIIRS>fusion) |
| Fig 3 | KLD barplot + JSD map | **New composite**: Posterior map + sensor-only + sensor precision (proves collapse is fusion artifact) |
| Fig 4 | Suppression map alone | **New composite**: KLD barplot + JSD map + suppression map (proves conflict, not absence) |
| Fig 5 | Consistency map + interval | **New composite**: Consistency map + interval comp + scatter (proves paradigms diverge) |
| Fig 6 | None | **New**: Conceptual model diagram (S2→structural, VIIRS→functional, conflict→collapse, future→multi-dim) |

All figures now use `width=0.95\textwidth` (was `0.48\textwidth` or `0.7\textwidth` in v1).

## Table Changes
- **v1**: 4 tables in body + 1 in appendix, formatting adequate.
- **v2**: Same 4+1 structure, but all numerics verified against source CSV files. Table 3 (KLD) now includes KL(Both || Both_global) ≈ 0 as the highlighted key diagnostic row. Tables use booktabs, no tabularx (JMLR class restriction).

## Formula Changes
- **v1**: Some equations had formatting issues in pandoc conversion.
- **v2**: All equations correctly typeset. Bayesian model separated into clear 3-level structure with explicit Level 1/2/3 labels. KLD and JSD equations use proper `\|` spacing.

## Number Verification
All key numbers are hardcoded (not macro-dependent) in v2:
- VIIRS AUC = 0.829, S2 AUC = 0.332 ✓
- tau = 0.063, sigma_s2 = 2.685, sigma_viirs = 0.904, delta_s2 = -1.705 ✓
- KL(Both || Both_global) ≈ 0, JSD = 0.1464, suppression = 46.8% ✓
- D-S Belief = 0.145, Plausibility = 0.369 ✓
- Genuine agreement = 49.9%, uncertainty-compatible = 10.2%, divergent = 28.1% ✓

## Wording Fixes
All prohibited phrases removed from v2:
- "VIIRS is better than S2" → "VIIRS shows stronger agreement with UNOSAT reference labels under this validation setting"
- "S2 is useless" → "the current S2-derived index is weakly aligned"
- "UNOSAT is ground truth" → "UNOSAT is an external reference rather than absolute ground truth"
- "Bayesian model failed" → "internally coherent but limited for spatial discrimination"
- "D-S is more correct" → removed entirely; framed as complementary paradigms
- "Bayesian is more accurate" → "Bayesian shrinkage is more consistent in some divergent grids relative to UNOSAT"

## Files Changed
- `paper/main.tex` → `paper/main_v2.tex` (complete rewrite)
- `paper/figures/fig*` → 6 new composite figures
- `paper/build/final_T20_paper_v2.pdf` — compiled and verified

## Known Issues
1. Author information still placeholder — replace `Author Name` and `author@university.edu`.
2. MiKTeX nag message "So far, you have not checked for MiKTeX updates" appears during compilation but does not affect output.
3. One `Label(s) may have changed` warning remains — run `pdflatex main_v2.tex` one more time to resolve.
