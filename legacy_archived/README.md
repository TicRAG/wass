# Legacy Archived Code

This folder contains the previous end-to-end experiment / training stack that has now been superseded by:

* `scripts/train_wass_paper_aligned.py` (PPO + Graph Encoder + RAG Teacher)
* `scripts/evaluate_paper_methods.py` (Unified evaluation & per-size analysis)

Archiving rules:
* Original files are moved here unchanged (only path relocation) so references in paper notes remain reproducible.
* New development should NOT import from this directory.
* If a legacy component is still needed, migrate the minimal logic into `src/` or `scripts/` with a comment.

Rationale:
The new pipeline provides decomposed reward logging, retrieval-augmented reward shaping, dynamic scheduling, and deterministic evaluation with per-workflow-size breakdowns. Retaining the old code without polluting the active namespace reduces maintenance risk and confusion.

If you need to restore a specific legacy module, copy it out instead of editing in place to keep the archive immutable.

Last updated: auto-archived on first migration.