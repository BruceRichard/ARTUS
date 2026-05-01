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

## Paper Figures

The project-level README references figures exported from `DecoArt.pdf`:

```text
docs/decoart/figure_1_overview_1.png
docs/decoart/figure_2_state_contact_1.png
docs/decoart/figure_3_routing_1.png
```
