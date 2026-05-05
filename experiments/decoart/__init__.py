"""DecoArt: Structured State Decoupling for Physically Valid Articulated Object Generation.

This package implements the evaluation-side quantities described in the NeurIPS 2026 paper:

- state_metrics.py:     Physical evaluation (E_tan, E_norm, E_pen, J_i) and representation
                        routing (T_str, T_phy, T_det) derived from the articulated structured state.
- build_state_metadata.py:  Batch processing CLI for computing per-object PV-Rate, physical
                            metrics, and routing assignments from generated outputs.
- compare_state_metrics.py: Delta comparison between no-guidance and guidance runs.

The physical evaluation branch (PEB) computes tangential alignment, normal consistency,
and non-penetration from bbox/joint/limit metadata without touching decoded meshes.
The representation routing branch (RRB) partitions part tokens into structure-preserving,
physically critical, and detail-oriented groups using depth, subtree size, and physical scores.
"""

