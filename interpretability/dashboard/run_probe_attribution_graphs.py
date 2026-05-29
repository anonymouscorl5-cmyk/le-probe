#!/usr/bin/env python3
"""
Precompute Neuronpedia-style attribution graph JSONs for workspace probe playbook entries.

After running, use recommend_neuronpedia_nodes.py on the output directory to get
which CLT features / patches to click in the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from interpretability.dashboard.probe_graph_runner import (
    generate_probe_graph,
    load_probe_resources,
)
from interpretability.dashboard.recommend_neuronpedia_nodes import (
    process_graph,
)


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


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
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-phase attribution timings (CUDA-synced) and IG step progress",
    )
    args = parser.parse_args()
    t0 = time.perf_counter()

    le_probe = Path(__file__).resolve().parents[2]
    playbook_path = Path(
        args.playbook
        or le_probe / "workspace_visualization/neuronpedia_probe_playbook.json"
    )
    _log(f"Stage 1/4: loading playbook from {playbook_path}")
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

    schemes = sorted({e.get("scheme", "?") for e in entries})
    _log(
        f"  → {len(entries)} probe(s) for variant={args.variant!r} "
        f"(schemes: {', '.join(schemes)})"
    )

    out_dir = Path(
        args.out_dir
        or le_probe / "workspace_visualization" / "attribution_graphs" / args.variant
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"  → output directory: {out_dir}")
    _log(f"Stage 1 done ({_fmt_elapsed(time.perf_counter() - t0)})")

    t_load = time.perf_counter()
    _log(
        f"Stage 2/4: loading {args.variant} checkpoint + transcoders + probe bundle "
        f"(device={args.device or 'auto'})…"
    )
    resources = load_probe_resources(args.variant, device=args.device, verbose=True)
    resources["variant"] = {"tag": args.variant}
    resources["profile"] = args.profile
    _log(
        f"Stage 2 done — model on {resources['device']}, "
        f"{len(resources['transcoders'])} CLT layer(s), "
        f"{len(resources['dataset'])} probes in bundle "
        f"({_fmt_elapsed(time.perf_counter() - t_load)})"
    )

    t_graphs = time.perf_counter()
    _log(f"Stage 3/4: running IG attribution on {len(entries)} probe(s)…")
    ok, failed = 0, 0
    bar = tqdm(
        entries,
        desc="Attribution graphs",
        unit="probe",
        file=sys.stdout,
        dynamic_ncols=True,
    )
    for entry in bar:
        pid = int(entry["probe_id"])
        scheme = entry.get("scheme", "na")
        cluster = entry.get("cluster", "na")
        role = entry.get("role", "canonical")
        fname = f"{scheme}_{cluster}_{role}_pid{pid}.json"
        out_path = out_dir / fname
        bar.set_postfix_str(f"pid={pid} {scheme}/{cluster}", refresh=False)

        try:
            t_probe = time.perf_counter()
            graph = generate_probe_graph(
                resources,
                bundle_index=int(entry["bundle_index"]),
                probe_id=pid,
                playbook_entry=entry,
            )
            out_path.write_text(json.dumps(graph, indent=2))
            n_nodes = len(graph.get("nodes", []))
            elapsed_probe = time.perf_counter() - t_probe

            if args.recommend:
                process_graph(out_path, quiet=True)

            ok += 1
            bar.write(
                f"  ✓ pid={pid} ({scheme}/{cluster}/{role}) — "
                f"{n_nodes} graph nodes, {_fmt_elapsed(elapsed_probe)}"
            )
        except Exception as exc:
            failed += 1
            bar.write(f"  ✗ pid={pid} ({scheme}/{cluster}/{role}) — {exc!r}")

    bar.close()
    _log(
        f"Stage 3 done — {ok} graph(s) written, {failed} failed "
        f"({_fmt_elapsed(time.perf_counter() - t_graphs)})"
    )

    _log(
        f"Stage 4/4: complete — total {_fmt_elapsed(time.perf_counter() - t0)}. "
        f"Graphs (+ .nodes.md if enabled) in {out_dir}"
    )
    if failed:
        raise SystemExit(f"{failed} probe(s) failed; see messages above.")
    _log(
        "Next: read *.nodes.md for encoder features to inspect, or open Neuronpedia locally."
    )


if __name__ == "__main__":
    main()
