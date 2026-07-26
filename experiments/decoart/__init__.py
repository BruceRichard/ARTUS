"""DecoArt: Structured State Decoupling for Physically Valid Articulated Object Generation.

This package implements the evaluation-side quantities described in the NeurIPS 2026 paper:

- state_metrics.py:     Physical evaluation (E_tan, E_norm, E_pen, J_i) and representation
                        routing (T_str, T_phy, T_det) derived from the articulated structured state.
- build_state_metadata.py:  Batch processing CLI for computing per-object PV-Rate, physical
                            metrics, and routing assignments from generated outputs.
- compare_state_metrics.py: Delta comparison between no-guidance and guidance runs.
- torch_guidance.py: Differentiable PEB/RRB and token-space validity-vector injection used
                     during every autoregressive tree-expansion round.
- make_rebuttal_configs.py / summarize_rebuttal_runs.py: Matched routing and perturbation
                     controls with five-repeat aggregation.
- mesh_proxy_correlation.py / pybullet_dynamic_eval.py: Decoded-mesh transfer analysis and
                     closed--open--closed dynamic validation.

The physical evaluation branch (PEB) computes tangential alignment, normal consistency,
and non-penetration from bbox/joint/limit metadata without touching decoded meshes.
The representation routing branch (RRB) partitions part tokens into structure-preserving,
physically critical, and detail-oriented groups using depth, subtree size, and physical scores.
"""
