# Paper v3 Changes from v2

## Compilation
- **v3**: 17 pages, 6.5 MB, 0 undefined citations, 0 label warnings, 0 errors.
- All references resolved (10 verified entries + 2 with verified DOIs added for Related Work).

## Structural Changes

### New: Section 2 - Related Work (~800 words)
- 2.1 Remote Sensing for Damage Assessment
- 2.2 Nighttime Lights and Functional Disruption
- 2.3 Bayesian Hierarchical Fusion
- 2.4 Dempster-Shafer Evidence Theory
- 2.5 Information-Theoretic Divergence for Sensor Comparison
- 10 references cited; no ? placeholders.

### New: Figure 1 - Evidence-chain workflow
- 6-node horizontal flowchart: A -> B -> C1 -> C2 -> C3 -> D
- UNOSAT validation anchor shown as cross-cutting reference
- Placed in Introduction, cross-referenced in Conclusion

### Renumbered Figures
| v2 | v3 | Content | Changes |
|----|----|---------|---------|
| Fig 1 | Fig 2 | Sensor disagreement | Same 3-panel, regenerated at 22 inch width |
| Fig 2 | Fig 3 | UNOSAT validation | Same ROC + AUC, regenerated at 20 inch width |
| Fig 3 | Fig 4 | Bayesian collapse | 3-panel, expanded from 21 to 22 inch width |
| Fig 4 (single) | Fig 5a + 5b | KLD/JSD | **SPLIT**: 5a = large barplot (14x5 in); 5b = JSD map + suppression map side by side (20x7.5 in) |
| Fig 5 (3-panel) | Fig 6 | Bayes vs DS | Simplified to 2 panels (consistency map + interval comparison) at 22x9 in; scatter plots moved to Appendix |
| Fig 6 (concept) | Fig 7 | Conceptual model | Unchanged, placed in Discussion 5.4 |

### Figure Sizing Improvements
- All figures now use `width=0.95\textwidth` (was 0.48--0.85 in v2)
- Maps no longer too small to read colorbars and spatial patterns
- KLD/JSD split into two separate figures (5a summary, 5b maps) because three dense panels couldn't fit legibly on one page
- Scatter plots moved to Appendix C to give consistency map and interval comparison more space

### Language Fixes
- "the only coherent Bayesian update" -> "a coherent response under this model specification"
- "D-S over-trusts noisy S2" -> "D-S may over-weight strong single-sensor evidence"
- "Bayesian is more accurate" removed entirely; replaced with "directionally consistent with the external reference in a majority of these cases"
- All instances of cautionary language preserved or strengthened

### References
- 12 entries total (10 verified with DOIs, 2 with confirmed publication data)
- Added: li2018 (Iraqi Civil War VIIRS, verified DOI), roman2018 fixed (now points to NASA Black Marble product suite paper)
- All TODO placeholders removed

### Appendix Expanded
- Appendix A: PR curves for UNOSAT validation (moved from body)
- Appendix B: Method comparison table (unchanged from v2)
- Appendix C: JSD vs divergence scatter plots (moved from body Figure 6)

## Files
| File | Description |
|------|-------------|
| `paper/main_v3.tex` | Complete LaTeX source |
| `paper/build/final_T20_paper_v3.pdf` | Compiled PDF (17 pages, 6.5 MB) |
| `paper/references.bib` | 12 verified bibliographic entries |
| `paper/figures/fig1_workflow.png` through `fig6_bayes_ds_v3.png` | 8 new/regenerated figures |
| `paper/README_v3_changes.md` | This file |

## Remaining Minor Issues
1. Author placeholder still present (`Author Name`, `author@university.edu`)
2. `\acks{}` still commented out
3. MiKTeX nag message about updates (cosmetic, does not affect output)
4. Some figure panels (especially Fig 2's maps) are resized from lower-resolution source PNGs; vector versions would be sharper
