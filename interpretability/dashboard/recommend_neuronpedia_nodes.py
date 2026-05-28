#!/usr/bin/env python3
"""
Read precomputed attribution graph JSONs and list prominent nodes to open in Neuronpedia.

Outputs a sidecar JSON + markdown brief per graph.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _layer_from_node_id(node_id: str) -> str | None:
    m = re.match(r"feat_(encoder_L\d+|predictor_L\d+)_\d+", node_id)
    return m.group(1) if m else None


def _parse_feature_layer(node: dict) -> tuple[str | None, int | None]:
    node_id = node.get("node_id", "")
    layer = _layer_from_node_id(node_id)
    feat = node.get("feature")
    if feat is not None and layer:
        return layer, int(feat)
    clerp = node.get("clerp", "")
    m = re.search(r"F(\d+) \((encoder_L\d+|predictor_L\d+)\)", clerp)
    if m:
        return m.group(2), int(m.group(1))
    return layer, int(feat) if feat is not None else None


def _is_encoder_layer(layer: str | None) -> bool:
    return bool(layer and layer.startswith("encoder_L"))


def rank_nodes(
    graph: dict,
    *,
    top_features: int = 12,
    top_inputs: int = 8,
    encoder_only: bool = True,
) -> dict:
    nodes = graph.get("nodes", [])
    meta = graph.get("metadata", {})
    diff_set = set(meta.get("differential_top_k") or [])

    features = []
    patches = []
    states = []
    logits = []

    for n in nodes:
        ft = n.get("feature_type", "")
        infl = abs(float(n.get("influence", 0.0)))
        entry = {
            "node_id": n.get("node_id"),
            "jsNodeId": n.get("jsNodeId", n.get("node_id")),
            "influence": float(n.get("influence", 0.0)),
            "abs_influence": infl,
            "clerp": n.get("clerp", ""),
            "streamIdx": n.get("streamIdx"),
            "token_prob": n.get("token_prob"),
        }
        if ft == "feature":
            layer, fid = _parse_feature_layer(n)
            entry["layer"] = layer
            entry["feature_id"] = fid
            entry["in_differential_top_k"] = (
                fid in diff_set if fid is not None else False
            )
            entry["neuronpedia_hint"] = (
                f"{layer} feature {fid}" if layer and fid is not None else None
            )
            features.append(entry)
        elif ft == "patch":
            patches.append(entry)
        elif ft == "state":
            states.append(entry)
        elif ft == "logit":
            logits.append(entry)

    features.sort(key=lambda x: x["abs_influence"], reverse=True)
    patches.sort(key=lambda x: x["abs_influence"], reverse=True)
    states.sort(key=lambda x: x["abs_influence"], reverse=True)

    predictor_features = [f for f in features if not _is_encoder_layer(f.get("layer"))]
    if encoder_only:
        features = [f for f in features if _is_encoder_layer(f.get("layer"))]

    # Prioritize features that match Tier-A differential list, then by |influence|
    features.sort(
        key=lambda x: (not x.get("in_differential_top_k"), -x["abs_influence"])
    )

    by_layer: dict[str, list[dict]] = {}
    for f in features:
        layer = f.get("layer") or "unknown"
        by_layer.setdefault(layer, []).append(f)

    unmatched_layers = {
        f.get("layer") for f in features if f.get("feature_id") is not None
    }
    diff_ref_layer = (
        "encoder_L0"
        if "encoder_L0" in unmatched_layers or not unmatched_layers
        else next(iter(sorted(unmatched_layers)), "encoder_L0")
    )

    return {
        "scope": {
            "encoder_only": encoder_only,
            "note": "Recommendations list encoder CLT features only; predictor/subgoal/DINO "
            "nodes may still appear in the full graph JSON.",
            "predictor_nodes_omitted": len(predictor_features) if encoder_only else 0,
        },
        "metadata": {
            k: meta.get(k)
            for k in (
                "probe_id",
                "bundle_index",
                "variant",
                "scheme",
                "cluster",
                "role",
                "differential_top_k",
            )
            if k in meta
        },
        "click_first": {
            "clt_features": features[:top_features],
            "input_patches": patches[:top_inputs],
            "state_dims": [] if encoder_only else states[:top_inputs],
            "target_logit": [] if encoder_only else logits[:1],
        },
        "by_layer": {
            layer: rows[:5]
            for layer, rows in sorted(by_layer.items())
            if not encoder_only or _is_encoder_layer(layer)
        },
        "differential_top_k_unmatched_in_graph": sorted(
            diff_set
            - {
                f["feature_id"]
                for f in features
                if f.get("feature_id") is not None and f.get("layer") == diff_ref_layer
            }
        )[:20],
    }


def render_markdown(rec: dict, graph_path: Path) -> str:
    meta = rec["metadata"]
    lines = [
        f"# Neuronpedia node picks — `{graph_path.name}`",
        "",
    ]
    if meta:
        lines.append("## Context")
        for k, v in meta.items():
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    scope = rec.get("scope", {})
    if scope.get("encoder_only"):
        lines.append(
            "> **Scope:** encoder CLT layers only (`encoder_L0`–`encoder_L11`). "
            "Predictor, reward/subgoal targets, and DINO branches in the graph JSON are ignored here."
        )
        lines.append("")
    lines.append("## Click these encoder CLT features first (IG-ranked)")
    lines.append("")
    lines.append("| Priority | Layer | Feature | |influence| | Tier-A diff? | Label |")
    lines.append("| --: | :-- | --: | --: | :-- | :-- |")
    for i, row in enumerate(rec["click_first"]["clt_features"], 1):
        lines.append(
            f"| {i} | `{row.get('layer', '?')}` | {row.get('feature_id', '?')} | "
            f"{row['abs_influence']:.4f} | "
            f"{'yes' if row.get('in_differential_top_k') else 'no'} | "
            f"{row.get('clerp', '')} |"
        )
    lines.append("")
    lines.append(
        "In the dashboard, search or jump to the **layer + feature index** above."
    )
    lines.append("")

    if rec["click_first"]["input_patches"]:
        lines.append("## Salient visual patches (input IG)")
        lines.append("")
        for row in rec["click_first"]["input_patches"][:6]:
            lines.append(
                f"- `{row['node_id']}` — {row['clerp']} (|inf|={row['abs_influence']:.4f})"
            )
        lines.append("")

    if rec["click_first"].get("state_dims"):
        lines.append("## Salient proprio / action dims")
        lines.append("")
        for row in rec["click_first"]["state_dims"][:6]:
            lines.append(
                f"- `{row['node_id']}` — {row['clerp']} (|inf|={row['abs_influence']:.4f})"
            )
        lines.append("")

    unmatched = rec.get("differential_top_k_unmatched_in_graph") or []
    if unmatched:
        lines.append("## Tier-A L0 features not in graph top features")
        lines.append("")
        lines.append(
            "These were cluster-differential on embeddings but did not appear among "
            "top IG nodes (try searching them manually on **encoder_L0**):"
        )
        lines.append("")
        lines.append(", ".join(str(x) for x in unmatched[:15]))
        lines.append("")

    lines.append("## Per-layer top picks")
    lines.append("")
    for layer, rows in rec.get("by_layer", {}).items():
        ids = [
            str(r.get("feature_id")) for r in rows if r.get("feature_id") is not None
        ]
        if ids:
            lines.append(f"- **{layer}:** {', '.join(ids)}")
    lines.append("")
    return "\n".join(lines)


def process_graph(
    path: Path,
    *,
    top_features: int = 12,
    top_inputs: int = 8,
    encoder_only: bool = True,
    quiet: bool = False,
) -> None:
    graph = json.loads(path.read_text())
    rec = rank_nodes(
        graph,
        top_features=top_features,
        top_inputs=top_inputs,
        encoder_only=encoder_only,
    )
    out_json = path.with_suffix(".nodes.json")
    out_md = path.with_suffix(".nodes.md")
    out_json.write_text(json.dumps(rec, indent=2))
    out_md.write_text(render_markdown(rec, path))
    if not quiet:
        print(f"Wrote {out_json}")
        print(f"Wrote {out_md}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend Neuronpedia nodes from attribution graph JSONs"
    )
    parser.add_argument(
        "graphs",
        nargs="+",
        help="Graph JSON file(s) or directory(ies) of *.json graphs",
    )
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--top-inputs", type=int, default=8)
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Include predictor features and action/reward nodes in recommendations",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for g in args.graphs:
        p = Path(g)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
            paths = [x for x in paths if not x.name.endswith(".nodes.json")]
        else:
            paths.append(p)

    for path in paths:
        if path.name.endswith(".nodes.json"):
            continue
        process_graph(
            path,
            top_features=args.top_features,
            top_inputs=args.top_inputs,
            encoder_only=not args.all_layers,
        )


if __name__ == "__main__":
    main()
