# count_params.py
# 用法:
# python count_params.py --ckpt path/to/model.ckpt
# python count_params.py --ckpt path/to/model.ckpt --module-prefix transformer
# python count_params.py --ckpt path/to/model.ckpt --group-prefixes transformer diffusion sdf

import argparse
import torch


def to_billion(n: int) -> float:
    return n / 1e9


def load_state_dict_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # 有些保存方式直接是参数字典
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise ValueError(f"无法从 {ckpt_path} 解析 state_dict")


def count_all_params(sd: dict):
    total = 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            total += v.numel()
    return total


def count_by_prefix(sd: dict, prefix: str):
    if not prefix.endswith("."):
        prefix = prefix + "."
    total = 0
    for k, v in sd.items():
        if k.startswith(prefix) and torch.is_tensor(v):
            total += v.numel()
    return total


def count_groups(sd: dict, prefixes):
    # prefixes: e.g. ["transformer", "diffusion", "sdf"]
    results = {}
    assigned = set()
    for p in prefixes:
        cnt = 0
        pp = p if p.endswith(".") else p + "."
        for k, v in sd.items():
            if k.startswith(pp) and torch.is_tensor(v):
                cnt += v.numel()
                assigned.add(k)
        results[p] = cnt

    other = 0
    for k, v in sd.items():
        if torch.is_tensor(v) and k not in assigned:
            other += v.numel()
    results["other"] = other
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="checkpoint 路径")
    parser.add_argument("--module-prefix", default=None, help="只统计某个前缀模块，如 transformer")
    parser.add_argument(
        "--group-prefixes",
        nargs="*",
        default=None,
        help="分组统计前缀，例如 transformer diffusion sdf",
    )
    args = parser.parse_args()

    sd = load_state_dict_from_ckpt(args.ckpt)

    total = count_all_params(sd)
    print(f"[TOTAL] {total:,} params  ({to_billion(total):.6f} B)")

    if args.module_prefix:
        m = count_by_prefix(sd, args.module_prefix)
        print(f"[{args.module_prefix}] {m:,} params  ({to_billion(m):.6f} B)")
        print(f"[{args.module_prefix}/TOTAL] {m/total:.4%}")

    if args.group_prefixes:
        res = count_groups(sd, args.group_prefixes)
        print("\n[GROUP BREAKDOWN]")
        for k, v in res.items():
            print(f"- {k}: {v:,} params  ({to_billion(v):.6f} B, {v/total:.4%})")


if __name__ == "__main__":
    main()
