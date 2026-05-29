#!/usr/bin/env python3
"""
Extract backward encoder circuits from precomputed Neuronpedia attribution graphs.

Writes sibling JSON files (e.g. foo.json -> foo.circuit.json) without modifying sources.
See docs/reports/2026-05-29_neuronpedia_cluster_highlights.md § Circuit extraction plan.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

ENCODER_NODE_RE = re.compile(r"^feat_encoder_L(\d+)_(\d+)$")


@dataclass
class ExtractParams:
    max_seeds: int = 6
    beam_width: int = 2
    max_nodes: int = 15
    min_edge_frac: float = 0.02
    cluster_prior_boost: float = 2.0
    encoder_top_layer: int = 11


@dataclass
class CircuitResult:
    node_ids: set[str] = field(default_factory=set)
    links: list[dict[str, Any]] = field(default_factory=list)
    seed_node_ids: list[str] = field(default_factory=list)
    depth_from_nearest_seed: dict[str, int] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)

    def link_key(self, link: dict[str, Any]) -> tuple[str, str, float]:
        return (link["source"], link["target"], float(link["weight"]))


def parse_encoder_layer(node_id: str) -> int | None:
    m = ENCODER_NODE_RE.match(node_id)
    return int(m.group(1)) if m else None


def is_encoder_node(node_id: str) -> bool:
    return node_id.startswith("feat_encoder_")


def build_incoming(links: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    incoming: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        incoming.setdefault(link["target"], []).append(link)
    return incoming


def order_pinned_node_ids(
    node_ids: list[str], seed_node_ids: list[str] | None = None
) -> list[str]:
    """L11-first display order; seeds listed before other nodes at the same layer."""
    seeds = seed_node_ids or []
    seed_rank = {nid: i for i, nid in enumerate(seeds)}

    def sort_key(nid: str) -> tuple[int, int, int, str]:
        layer = parse_encoder_layer(nid)
        layer_num = layer if layer is not None else -1
        return (
            0 if nid in seed_rank else 1,
            seed_rank.get(nid, 999),
            -layer_num,
            nid,
        )

    return sorted(node_ids, key=sort_key)


def build_neuronpedia_graph_url(
    slug: str,
    pinned_ids: list[str],
    *,
    seed_node_ids: list[str] | None = None,
    base_url: str = "http://localhost:3000",
    model_id: str = "lewm-robot",
    pruning_threshold: int | float = 99999,
    density_threshold: float = 0.99,
) -> str:
    """Open graph with circuit nodes pre-pinned (matches Neuronpedia UI query params)."""
    ordered = order_pinned_node_ids(pinned_ids, seed_node_ids)
    query = urlencode(
        {
            "slug": slug,
            "pruningThreshold": str(int(pruning_threshold)),
            "densityThreshold": str(density_threshold),
            "pinnedIds": ",".join(ordered),
        }
    )
    return f"{base_url.rstrip('/')}/{model_id}/graph?{query}"


def circuit_slug_from_path(circuit_path: Path) -> str:
    name = circuit_path.name
    if name.endswith(".circuit.json"):
        return name[: -len(".circuit.json")]
    return circuit_path.stem


def neuronpedia_url_from_circuit(
    circuit: dict[str, Any],
    circuit_path: Path | None = None,
    **url_kwargs: Any,
) -> str:
    slug = circuit_slug_from_path(circuit_path) if circuit_path else ""
    if not slug:
        meta = circuit.get("metadata") or {}
        slug = str(meta.get("variant", ""))
        parts = [
            meta.get("scheme"),
            meta.get("cluster"),
            meta.get("role"),
            f"pid{meta.get('probe_id')}",
        ]
        slug = "_".join(str(p) for p in parts if p)
    extraction = (circuit.get("metadata") or {}).get("circuit_extraction") or {}
    pinned = extraction.get("pinned_ids") or [
        n["node_id"] for n in circuit.get("nodes", [])
    ]
    seeds = (
        extraction.get("seed_node_ids")
        or (circuit.get("circuit") or {}).get("seed_node_ids")
        or []
    )
    meta = circuit.get("metadata") or {}
    prune = meta.get("node_threshold", 99999)
    return build_neuronpedia_graph_url(
        slug, pinned, seed_node_ids=seeds, pruning_threshold=prune, **url_kwargs
    )


def cluster_prior_set(metadata: dict[str, Any]) -> set[int]:
    raw = metadata.get("differential_top_k") or []
    return {int(x) for x in raw}


def score_incoming_edge(
    link: dict[str, Any],
    nodes_by_id: dict[str, dict[str, Any]],
    prior_features: set[int],
    prior_boost: float,
) -> float:
    parent = nodes_by_id.get(link["source"])
    if not parent:
        return 0.0
    weight = float(link.get("weight") or 0.0)
    influence = float(parent.get("influence") or 0.0)
    fid = int(parent.get("feature", -1))
    prior = prior_boost if fid in prior_features else 1.0
    return weight * math.log1p(max(influence, 0.0)) * prior


def select_seeds(
    graph: dict[str, Any],
    incoming: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
    params: ExtractParams,
    prior_features: set[int],
) -> list[str]:
    """Union: differential L11, top logit parents at L11, top influence at L11."""
    l11_nodes = [
        n
        for n in graph.get("nodes", [])
        if is_encoder_node(n["node_id"])
        and parse_encoder_layer(n["node_id"]) == params.encoder_top_layer
    ]
    if not l11_nodes:
        return []

    seeds: list[str] = []
    seen: set[str] = set()

    def add(node_id: str) -> None:
        if node_id not in seen and len(seeds) < params.max_seeds:
            seen.add(node_id)
            seeds.append(node_id)

    # A: cluster-enriched (differential_top_k) at L11
    for n in sorted(
        l11_nodes, key=lambda x: float(x.get("influence") or 0), reverse=True
    ):
        if int(n.get("feature", -1)) in prior_features:
            add(n["node_id"])

    # B: top incoming weight to logit_success
    logit_id = "logit_success"
    for link in sorted(
        incoming.get(logit_id, []),
        key=lambda l: float(l.get("weight") or 0),
        reverse=True,
    ):
        if (
            is_encoder_node(link["source"])
            and parse_encoder_layer(link["source"]) == params.encoder_top_layer
        ):
            add(link["source"])

    # C: top influence at L11
    for n in sorted(
        l11_nodes, key=lambda x: float(x.get("influence") or 0), reverse=True
    ):
        add(n["node_id"])

    return seeds[: params.max_seeds]


def expand_seed_forest(
    seed_id: str,
    incoming: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
    prior_features: set[int],
    params: ExtractParams,
) -> tuple[set[str], list[dict[str, Any]]]:
    """Backward forest from one seed; stop expanding at L0."""
    circuit_nodes: set[str] = {seed_id}
    link_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    expanded: set[str] = set()
    queue: deque[str] = deque([seed_id])

    while queue and len(circuit_nodes) < params.max_nodes:
        nid = queue.popleft()
        if nid in expanded:
            continue
        expanded.add(nid)

        layer = parse_encoder_layer(nid)
        if layer is None or layer == 0:
            continue

        candidates: list[tuple[float, dict[str, Any]]] = []
        for link in incoming.get(nid, []):
            src = link["source"]
            if not is_encoder_node(src):
                continue
            src_layer = parse_encoder_layer(src)
            if src_layer is None or src_layer >= layer:
                continue
            s = score_incoming_edge(
                link, nodes_by_id, prior_features, params.cluster_prior_boost
            )
            if s <= 0:
                continue
            candidates.append((s, link))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        max_score = candidates[0][0]
        min_score = max_score * params.min_edge_frac

        for score, link in candidates[: params.beam_width]:
            if score < min_score:
                continue
            src = link["source"]
            link_by_key[(link["source"], link["target"], float(link["weight"]))] = link
            circuit_nodes.add(src)
            if len(circuit_nodes) >= params.max_nodes:
                break
            src_layer = parse_encoder_layer(src)
            if src_layer is not None and src_layer > 0 and src not in expanded:
                queue.append(src)

    return circuit_nodes, list(link_by_key.values())


def node_importance_score(
    node_id: str,
    links: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    seed_ids: set[str],
    prior_features: set[int],
    prior_boost: float,
) -> float:
    if node_id in seed_ids:
        return 1e12
    score = 0.0
    for link in links:
        if link["source"] == node_id or link["target"] == node_id:
            score += float(link.get("weight") or 0.0)
    node = nodes_by_id.get(node_id, {})
    score += math.log1p(max(float(node.get("influence") or 0.0), 0.0))
    fid = int(node.get("feature", -1))
    if fid in prior_features:
        score *= prior_boost
    depth_penalty = 0.0
    layer = parse_encoder_layer(node_id)
    if layer is not None:
        depth_penalty = layer * 0.01
    return score - depth_penalty


def prune_circuit(
    result: CircuitResult,
    nodes_by_id: dict[str, dict[str, Any]],
    prior_features: set[int],
    params: ExtractParams,
) -> tuple[int, bool]:
    """Hard cap on nodes; always keep seeds. Returns (n_before, was_pruned)."""
    n_before = len(result.node_ids)
    if n_before <= params.max_nodes:
        return n_before, False

    seed_set = set(result.seed_node_ids)
    ranked = sorted(
        result.node_ids,
        key=lambda nid: node_importance_score(
            nid,
            result.links,
            nodes_by_id,
            seed_set,
            prior_features,
            params.cluster_prior_boost,
        ),
        reverse=True,
    )
    keep: set[str] = set()
    for nid in ranked:
        if len(keep) >= params.max_nodes:
            break
        keep.add(nid)
    for nid in seed_set:
        if nid in result.node_ids:
            keep.add(nid)
    if len(keep) > params.max_nodes:
        ranked_seeds_first = sorted(
            keep,
            key=lambda nid: node_importance_score(
                nid,
                result.links,
                nodes_by_id,
                seed_set,
                prior_features,
                params.cluster_prior_boost,
            ),
            reverse=True,
        )
        keep = set(ranked_seeds_first[: params.max_nodes])

    result.node_ids = keep
    result.links = [
        link
        for link in result.links
        if link["source"] in keep and link["target"] in keep
    ]
    result.depth_from_nearest_seed = {
        k: v for k, v in result.depth_from_nearest_seed.items() if k in keep
    }
    return n_before, True


def merge_forests(
    seed_ids: list[str],
    incoming: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
    prior_features: set[int],
    params: ExtractParams,
) -> CircuitResult:
    result = CircuitResult(seed_node_ids=list(seed_ids))
    all_links: dict[tuple[str, str, float], dict[str, Any]] = {}

    for seed in seed_ids:
        if seed not in nodes_by_id:
            continue
        remaining = params.max_nodes - len(result.node_ids)
        if remaining <= 0:
            break
        seed_params = ExtractParams(
            max_seeds=params.max_seeds,
            beam_width=params.beam_width,
            max_nodes=max(remaining, 1),
            min_edge_frac=params.min_edge_frac,
            cluster_prior_boost=params.cluster_prior_boost,
            encoder_top_layer=params.encoder_top_layer,
        )
        nodes, links = expand_seed_forest(
            seed, incoming, nodes_by_id, prior_features, seed_params
        )
        result.node_ids |= nodes
        for link in links:
            all_links[result.link_key(link)] = link
        for nid in nodes:
            result.depth_from_nearest_seed.setdefault(nid, 10**9)
        # BFS depth from this seed (backward distance)
        depth_q: deque[tuple[str, int]] = deque([(seed, 0)])
        seen_depth: set[str] = {seed}
        while depth_q:
            nid, d = depth_q.popleft()
            if d < result.depth_from_nearest_seed.get(nid, 10**9):
                result.depth_from_nearest_seed[nid] = d
            layer = parse_encoder_layer(nid)
            if layer is None or layer == 0:
                continue
            for link in all_links.values():
                if link["target"] != nid:
                    continue
                src = link["source"]
                if src not in nodes or not is_encoder_node(src):
                    continue
                src_layer = parse_encoder_layer(src)
                if src_layer is None or src_layer >= (layer or 0):
                    continue
                if src not in seen_depth:
                    seen_depth.add(src)
                    depth_q.append((src, d + 1))

    result.links = list(all_links.values())
    return result


def format_node_short(node_id: str, nodes_by_id: dict[str, dict[str, Any]]) -> str:
    layer = parse_encoder_layer(node_id)
    if layer is not None:
        fid = nodes_by_id.get(node_id, {}).get("feature", "?")
        return f"L{layer}:{fid}"
    return node_id


def build_paths(
    seed_ids: list[str],
    result: CircuitResult,
    incoming: dict[str, list[dict[str, Any]]],
    nodes_by_id: dict[str, dict[str, Any]],
    max_paths_per_seed: int = 8,
    max_path_len: int = 14,
) -> list[str]:
    """Enumerate backward paths seed -> ... -> L0 leaves (bounded)."""
    inc_in_circuit: dict[str, list[str]] = {}
    for link in result.links:
        inc_in_circuit.setdefault(link["target"], []).append(link["source"])

    paths: list[str] = []

    def dfs(current: str, chain: list[str], seed: str) -> None:
        if len(paths) >= max_paths_per_seed * len(seed_ids):
            return
        layer = parse_encoder_layer(current)
        if layer == 0 or not inc_in_circuit.get(current):
            paths.append(" <- ".join(chain))
            return
        if len(chain) >= max_path_len:
            paths.append(" <- ".join(chain) + " …")
            return
        for parent in sorted(
            inc_in_circuit.get(current, []),
            key=lambda p: float(nodes_by_id.get(p, {}).get("influence") or 0),
            reverse=True,
        )[:3]:
            if parent in chain:
                continue
            dfs(parent, chain + [format_node_short(parent, nodes_by_id)], seed)

    for seed in seed_ids:
        seed_paths_before = len(paths)
        dfs(seed, [format_node_short(seed, nodes_by_id)], seed)
        if len(paths) == seed_paths_before:
            paths.append(format_node_short(seed, nodes_by_id))

    return paths[: max_paths_per_seed * max(1, len(seed_ids))]


def skip_layer_stats(
    links: list[dict[str, Any]], nodes_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    skip = 0
    total = 0
    for link in links:
        sl = parse_encoder_layer(link["source"])
        tl = parse_encoder_layer(link["target"])
        if sl is None or tl is None:
            continue
        total += 1
        if tl - sl > 1:
            skip += 1
    return {
        "encoder_encoder_edges": total,
        "skip_layer_edges": skip,
        "skip_layer_fraction": (skip / total) if total else 0.0,
    }


def extract_circuit_from_graph(
    graph: dict[str, Any],
    params: ExtractParams,
    graph_slug: str,
) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    nodes_by_id = {n["node_id"]: n for n in nodes}
    incoming = build_incoming(links)
    meta = dict(graph.get("metadata") or {})
    prior_features = cluster_prior_set(meta)

    seeds = select_seeds(graph, incoming, nodes_by_id, params, prior_features)
    seed_cap = max(1, params.max_nodes // 2)
    if len(seeds) > seed_cap:
        seeds = seeds[:seed_cap]
    if not seeds:
        # Fallback: strongest L10/L11 encoder by influence
        enc = [
            n
            for n in nodes
            if is_encoder_node(n["node_id"])
            and (parse_encoder_layer(n["node_id"]) or -1)
            >= params.encoder_top_layer - 1
        ]
        enc.sort(key=lambda n: float(n.get("influence") or 0), reverse=True)
        seeds = [n["node_id"] for n in enc[: params.max_seeds]]

    result = merge_forests(seeds, incoming, nodes_by_id, prior_features, params)
    n_before_prune, was_pruned = prune_circuit(
        result, nodes_by_id, prior_features, params
    )
    result.seed_node_ids = [s for s in result.seed_node_ids if s in result.node_ids]
    result.paths = build_paths(result.seed_node_ids, result, incoming, nodes_by_id)

    circuit_nodes = [
        nodes_by_id[nid] for nid in sorted(result.node_ids) if nid in nodes_by_id
    ]
    circuit_links = result.links

    pinned_sorted = order_pinned_node_ids(list(result.node_ids), result.seed_node_ids)
    neuronpedia_url = build_neuronpedia_graph_url(
        graph_slug,
        pinned_sorted,
        seed_node_ids=result.seed_node_ids,
        pruning_threshold=meta.get("node_threshold", 99999),
    )

    out_meta = {**meta}
    out_meta["circuit_extraction"] = {
        "version": 2,
        "algorithm": "backward_encoder_forest",
        "seed_node_ids": result.seed_node_ids,
        "params": {
            "max_seeds": params.max_seeds,
            "beam_width": params.beam_width,
            "max_nodes": params.max_nodes,
            "min_edge_frac": params.min_edge_frac,
            "cluster_prior_boost": params.cluster_prior_boost,
            "encoder_top_layer": params.encoder_top_layer,
        },
        "n_nodes_before_prune": n_before_prune,
        "pruned_to_max_nodes": was_pruned,
        "n_nodes": len(circuit_nodes),
        "n_links": len(circuit_links),
        "differential_top_k_used": sorted(prior_features),
        "skip_layer_stats": skip_layer_stats(circuit_links, nodes_by_id),
        "paths_sample": result.paths[:20],
        "pinned_ids": pinned_sorted,
        "neuronpedia_url": neuronpedia_url,
    }

    return {
        "metadata": out_meta,
        "qParams": graph.get("qParams") or {},
        "nodes": circuit_nodes,
        "links": circuit_links,
        "circuit": {
            "seed_node_ids": result.seed_node_ids,
            "depth_from_nearest_seed": result.depth_from_nearest_seed,
            "paths": result.paths,
        },
    }


def circuit_output_path(input_path: Path, output_dir: Path | None) -> Path:
    stem = (
        input_path.name[: -len(".json")]
        if input_path.name.endswith(".json")
        else input_path.name
    )
    out_dir = output_dir if output_dir is not None else input_path.parent / "circuits"
    return out_dir / f"{stem}.circuit.json"


def iter_graph_jsons(input_dir: Path, variant_prefix: str | None) -> list[Path]:
    files = sorted(input_dir.glob("*.json"))
    out: list[Path] = []
    for p in files:
        if p.name.endswith(".nodes.json") or p.name.endswith(".circuit.json"):
            continue
        if variant_prefix:
            if variant_prefix == "multiview":
                if not p.name.startswith("multiview_") or p.name.startswith(
                    "multiview_skeleton"
                ):
                    continue
            elif variant_prefix == "multiview_skeleton":
                if not p.name.startswith("multiview_skeleton_") or p.name.startswith(
                    "multiview_skeleton_dino_"
                ):
                    continue
            elif not p.name.startswith(f"{variant_prefix}_"):
                continue
        out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract encoder circuits from attribution graph JSONs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of full attribution graphs (*.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-dir>/circuits)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Only process files whose name starts with <variant>_ (e.g. multiview_skeleton)",
    )
    parser.add_argument("--max-seeds", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=15,
        help="Hard cap on encoder nodes per circuit (default 15)",
    )
    parser.add_argument("--min-edge-frac", type=float, default=0.02)
    parser.add_argument(
        "--limit", type=int, default=0, help="Process at most N graphs (0 = all)"
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not (input_dir / "circuits").exists():
        (input_dir / "circuits").mkdir(parents=True, exist_ok=True)

    params = ExtractParams(
        max_seeds=args.max_seeds,
        beam_width=args.beam_width,
        max_nodes=args.max_nodes,
        min_edge_frac=args.min_edge_frac,
    )

    paths = iter_graph_jsons(input_dir, args.variant)
    if args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        raise SystemExit(f"No graph JSON files found in {input_dir}")

    print(f"Extracting circuits from {len(paths)} graph(s) in {input_dir}")
    for i, path in enumerate(paths, 1):
        graph = json.loads(path.read_text())
        graph_slug = path.stem
        circuit = extract_circuit_from_graph(graph, params, graph_slug)
        out_path = circuit_output_path(path, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(circuit, indent=2))
        ce = circuit["metadata"]["circuit_extraction"]
        print(
            f"  [{i}/{len(paths)}] {path.name} -> {out_path.name} "
            f"({ce['n_nodes']} nodes, {ce['n_links']} links, {len(ce['seed_node_ids'])} seeds)"
        )

    dest = output_dir or (input_dir / "circuits")
    print(f"Done. Circuit JSONs written to {dest}")


if __name__ == "__main__":
    main()
