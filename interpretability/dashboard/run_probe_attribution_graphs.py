#!/usr/bin/env python3
"""
Precompute Neuronpedia-style attribution graph JSONs for workspace probe playbook entries.

After running, use recommend_neuronpedia_nodes.py on the output directory to get
which CLT features / patches to click in the dashboard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interpretability.dashboard.probe_graph_runner import (
    generate_probe_graph,
    load_probe_resources,
)
from interpretability.dashboard.recommend_neuronpedia_nodes import (
    process_graph,
)


def load_playbook(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        return doc
    return doc.get("entries", [])


def filter_entries(
    entries: list[dict],
    *,
    variant: str | None,
    scheme: str | None,
    role: str | None,
    probe_ids: list[int] | None,
    limit: int | None,
) -> list[dict]:
    out = entries
    if variant:
        out = [e for e in out if e.get("variant") == variant]
    if scheme:
        out = [e for e in out if e.get("scheme") == scheme]
    if role:
        out = [e for e in out if e.get("role") == role]
    if probe_ids:
        want = set(probe_ids)
        out = [e for e in out if int(e["probe_id"]) in want]
    if limit is not None:
        out = out[:limit]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--playbook",
        type=str,
        default=None,
        help="neuronpedia_probe_playbook.json (default: workspace_visualization/...)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant tag from variant_profiles.yaml",
    )
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument(
        "--role", type=str, default=None, choices=("canonical", "borderline")
    )
    parser.add_argument("--probe-id", type=int, action="append", dest="probe_ids")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for graph JSONs",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--recommend",
        action="store_true",
        default=True,
        help="Write .nodes.json / .nodes.md sidecars (default: on)",
    )
    parser.add_argument(
        "--no-recommend",
        action="store_false",
        dest="recommend",
        help="Skip node recommendation sidecars",
    )
    args = parser.parse_args()

    le_probe = Path(__file__).resolve().parents[2]
    playbook_path = Path(
        args.playbook
        or le_probe / "workspace_visualization/neuronpedia_probe_playbook.json"
    )
    if not playbook_path.exists():
        raise FileNotFoundError(
            f"Playbook not found: {playbook_path}\n"
            "Run: python interpretability/transcoders/build_neuronpedia_probe_playbook.py"
        )

    entries = filter_entries(
        load_playbook(playbook_path),
        variant=args.variant,
        scheme=args.scheme,
        role=args.role,
        probe_ids=args.probe_ids,
        limit=args.limit,
    )
    if not entries:
        raise SystemExit("No playbook entries matched filters.")

    out_dir = Path(
        args.out_dir
        or le_probe / "workspace_visualization" / "attribution_graphs" / args.variant
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.variant} model + transcoders...")
    resources = load_probe_resources(args.variant, device=args.device)
    resources["variant"] = {"tag": args.variant}

    for i, entry in enumerate(entries):
        pid = int(entry["probe_id"])
        scheme = entry.get("scheme", "na")
        cluster = entry.get("cluster", "na")
        role = entry.get("role", "canonical")
        fname = f"{scheme}_{cluster}_{role}_pid{pid}.json"
        out_path = out_dir / fname

        print(f"[{i+1}/{len(entries)}] probe {pid} ({scheme}/{cluster}/{role})")
        graph = generate_probe_graph(
            resources,
            bundle_index=int(entry["bundle_index"]),
            probe_id=pid,
            playbook_entry=entry,
        )
        out_path.write_text(json.dumps(graph, indent=2))

        if args.recommend:
            process_graph(out_path)

    print(f"Done. Graphs in {out_dir}")
    print(
        "Open Neuronpedia with the matching variant engine, then use each "
        "*.nodes.md file for which features to inspect."
    )


if __name__ == "__main__":
    main()
