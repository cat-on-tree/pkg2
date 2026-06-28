## Current status

Stage 01 graph construction is complete.

Current official algorithm prototype graph:

```text
data/processed/patent_centered_subgraph_2018_2019_diabetes_conservative_v2_aggregated/
```

The project is now entering Stage 02:

```text
link prediction dataset construction and baseline algorithm experiments
```

## Engineering rules for Stage 02

- Do not refactor scripts 01-10 unless explicitly requested.
- Treat scripts 01-10 as completed graph construction pipeline scripts.
- From script 11 onward, put reusable logic under `src/pkg2/`.
- Keep `scripts/` as thin CLI orchestration layers.
- Do not overwrite existing processed graph outputs unless explicitly requested.
- Prefer creating versioned output directories.
- The first recommended task is Patent-Paper link prediction.
- Build both full-context and no-species dataset variants when possible.