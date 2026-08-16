# Statistical Drift Detection Thesis

Manuscript repository for the thesis *Statistical Drift Detection for Non-Stationary Black-Box LLM Classifiers*.

This repository owns the manuscript source, the publication-ready figures and tables consumed by the LaTeX build, and the small sync/render scripts that turn frozen evidence into thesis assets. The public code and reproducibility pipeline lives in the separate repository [`GeorgiiBurdinComo/llm-batch-eval`](https://github.com/GeorgiiBurdinComo/llm-batch-eval).

## Repository boundary

This repository contains:

- `thesis.tex` and `chapters/` for the manuscript source
- `literature.bib` and figure `.tex` sources
- `assets/evidence/` and `assets/metrics_export/` as committed thesis build inputs
- `scripts/` for evidence-sync and figure/table regeneration used by the manuscript

This repository does not need temporary audit screenshots, editor metadata, local caches, or duplicate output files for publication.
The public branch excludes historical screenshot clutter and other scratch-image variants that are not cited by the manuscript.

## Reproducibility assets

The thesis build consumes committed publication-ready assets from this repository, including:

- rendered figures under `assets/evidence/`
- generated tables/macros under `assets/metrics_export/`
- prompt excerpts copied into the thesis asset tree

The upstream pipeline code, frozen Langfuse CSV snapshots, prompt-optimisation splits, and canonical evidence export remain in [`GeorgiiBurdinComo/llm-batch-eval`](https://github.com/GeorgiiBurdinComo/llm-batch-eval).

## Build

The project uses `latexmk` with outputs written to `build/`.

```bash
latexmk -pdf thesis.tex
```

## Important scripts

- `scripts/sync_canonical_evidence.py` copies frozen evidence and prompt artifacts into the thesis asset tree
- `scripts/compute_planning_power.py` and `scripts/mcnemar_power.py` support the statistical figures and tables
- `scripts/regen_*.py` rebuild selected publication figures from committed source data

If the code/repro repository is not checked out as a sibling directory named `benchmark_eval`, set `BENCHMARK_EVAL_ROOT` before running `scripts/sync_canonical_evidence.py`.
