# DecoArt Experiment Code

This folder contains experiment-side code for the DecoArt paper. It computes physical validity and representation routing quantities directly from ArtFormer/DecoArt structured metadata.

## What It Computes

For each articulated part `i`, the code derives contact samples from the child box, support samples from the parent or nearby boxes, and computes:

```text
E_norm = 1 + n^T m
E_pen  = -n^T(q - s) - delta_i
E_tan  = ||(I - nn^T)(q - s)||_2
```

The part-wise physical cost follows the paper's inference-time state cost:

```text
J_i = -Avg[ rho * exp(-(E_tan + alpha * E_norm) / dmax) ]
      + beta * Avg[ softplus(E_pen / tau_pen) ]
```

It also computes representation routing scores:

```text
r_str  = lambda_depth * (1 - depth(i) / max_depth)
         + lambda_subtree * |V_i| / N

r_phys = lambda_delta * normalized(E_tan)
         + lambda_omega * normalized([E_pen]_+)
```

Each part is routed into one of:

```text
structure | physical | detail
```

## Inputs

Supported inputs:

```text
data/datasets/4_transformer_dataset/*.json
data/datasets/1_preprocessed_info/*.json
elog/.../output.dat
```

For `4_transformer_dataset` and generated `output.dat`, use the default box format:

```text
--bbox-format center_size
```

For raw `1_preprocessed_info`, use:

```text
--bbox-format min_max
```

## Build Metrics

Transformer metadata:

```bash
python experiments/decoart/build_state_metadata.py \
  --input data/datasets/4_transformer_dataset \
  --pattern "*.json" \
  --output-dir experiments/decoart/outputs/train_state \
  --bbox-format center_size
```

Generated outputs:

```bash
python experiments/decoart/build_state_metadata.py \
  --input elog/final_output/ours_Table \
  --pattern "output.dat" \
  --output-dir experiments/decoart/outputs/ours_table \
  --bbox-format center_size
```

The script writes:

```text
decoart_state_metrics.jsonl
decoart_part_metrics.csv
decoart_summary.json
```

## Compare No-Guidance vs Guidance

```bash
python experiments/decoart/compare_state_metrics.py \
  --baseline experiments/decoart/outputs/no_guidance/decoart_state_metrics.jsonl \
  --ours experiments/decoart/outputs/guidance/decoart_state_metrics.jsonl \
  --output experiments/decoart/outputs/guidance_delta.json
```

Use the resulting `pv_rate`, `align`, `normal`, `penetration`, and `j_cost` values in the paper tables.

## Inference-time Structured Guidance

The executable Eq. (13) implementation is in `torch_guidance.py` and is
called from `model/Transformer/eval/__init__.py` after every tree-expansion
round. Its code-grounded validity vector is

```text
v_i_phys = -d J_i(D_state(e_i)) / d e_i
```

The default fixed intervention coefficients are:

```text
[omega_str, omega_phy, omega_det] = [0.50, 1.00, 0.25]
```

The direction is normalized before applying the cost severity and routing
coefficient. A backtracking step accepts an intervention only when the
evaluated part-wise physical cost does not increase.

## Rebuttal Controls

Generate matched configs for correct/random/uniform routing, normalized state
noise at 1%/3%/5%, and routing/support corruption at 10%/20%/30%:

```bash
python experiments/decoart/make_rebuttal_configs.py
```

Run every generated config with the same object set and per-sample seeds:

```bash
for config in configs/3_TF-Diff/rebuttal/*.yaml; do
  python 3_pred_trans.py -c "$config"
done
```

Aggregate five-repeat PV-Rate, physical errors, routing counts, and paired
pre/post-intervention costs:

```bash
python experiments/decoart/summarize_rebuttal_runs.py \
  --root elog/final_output \
  --conditions routing_correct routing_random routing_uniform \
  --output experiments/decoart/outputs/rebuttal_summary.json
```

Evaluate decoded meshes and calculate the state-to-mesh Spearman correlations:

```bash
python experiments/decoart/mesh_proxy_correlation.py \
  --input elog/final_output/routing_correct \
  --output experiments/decoart/outputs/mesh_state_correlation.json
```

Run closed--open--closed dynamic validation:

```bash
python experiments/decoart/pybullet_dynamic_eval.py \
  --input elog/final_output/routing_correct \
  --output experiments/decoart/outputs/dynamic_routing_correct.json
```

Repeat the last command for stress-condition directories to obtain their
corresponding Dyn-SR. The dynamic JSON can also be passed to
`mesh_proxy_correlation.py --dynamic-results` to compare the mean state cost
of PyBullet-successful and failed objects.

## Paper Figures

The project-level README references figures exported from `DecoArt.pdf`:

```text
docs/decoart/figure_1_overview_1.png
docs/decoart/figure_2_state_contact_1.png
docs/decoart/figure_3_routing_1.png
```
