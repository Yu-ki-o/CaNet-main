# Research Memory

This directory is the long-lived research memory for the CaNet/front-door graph OOD project.

It is designed for incremental updates. New papers should be added as new `papers/Pxxx_*.md` files, registered in `meta/paper_registry.csv`, and then merged into the topic index, method cards, and idea queue without deleting prior insights.

## Layout

- `index.md`: cross-paper knowledge map and current research priorities.
- `project_context.md`: project-specific understanding of CaNet, front-door variants, and the current repo.
- `papers/`: one structured paper card per paper.
- `methods/`: reusable method cards distilled across papers.
- `ideas/`: implementable model variants and experiment queue.
- `experiments/`: protocols and dataset notes tied to this repo.
- `meta/`: registry, update log, taxonomy, and the prompt used to maintain this memory.

## Update Rule

When adding papers later:

1. Check `meta/paper_registry.csv` for duplicates.
2. Assign the next stable paper id, such as `P007`.
3. Create one paper card under `papers/`.
4. Register the paper in `meta/paper_registry.csv`.
5. Update only the relevant parts of `index.md`, `methods/`, and `ideas/`.
6. Record the change in `meta/update_log.md`.

All claims in method cards and ideas should cite paper ids. If a new paper conflicts with older notes, keep both and add a `Conflict / Reconciliation` note.
