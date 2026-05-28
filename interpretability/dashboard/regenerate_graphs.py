import argparse
import json
import os
from pathlib import Path

import requests

from interpretability.dashboard.probe_graph_runner import (
    generate_probe_graph,
    load_probe_resources,
)
from interpretability.dashboard.recommend_neuronpedia_nodes import process_graph


def regenerate_training_via_api(engine_url: str, output_subdir: str = ""):
    print(f"🚀 Using Engine API at: {engine_url}")

    scenarios = [
        {"slug": "grasp-success-1", "index": 452, "joint": 7},
        {"slug": "grasp-fail-1", "index": 120, "joint": 7},
        {"slug": "approach-can", "index": 800, "joint": 7},
        {"slug": "pre-grasp-pos", "index": 300, "joint": 7},
    ]

    base = os.path.join(
        os.getcwd(),
        "interpretability/neuronpedia/apps/webapp/public/graphs/lewm-robot",
    )
    out_dir = os.path.join(base, output_subdir) if output_subdir else base
    os.makedirs(out_dir, exist_ok=True)

    for scene in scenarios:
        print(
            f"📊 Fetching attribution for {scene['slug']} (index {scene['index']})..."
        )
        endpoint = f"{engine_url.rstrip('/')}/api/attribution/generate-graph"
        payload = {"prompt": f"{scene['index']}:{scene['joint']}"}
        try:
            response = requests.post(endpoint, json=payload, timeout=120)
            response.raise_for_status()
            graph_data = response.json()
            file_path = os.path.join(out_dir, f"{scene['slug']}.json")
            with open(file_path, "w") as f:
                json.dump(graph_data, f, indent=2)
            print(f"✅ Saved to {file_path}")
        except Exception as e:
            print(f"❌ Failed to fetch {scene['slug']}: {e}")


def load_playbook_entries(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        return doc
    return doc.get("entries", [])


def regenerate_playbook_local(
    playbook_path: Path,
    variant: str,
    out_dir: Path,
    *,
    scheme: str | None = None,
    role: str | None = None,
    limit: int | None = None,
    recommend: bool = True,
    device: str | None = None,
):
    entries = load_playbook_entries(playbook_path)
    if variant:
        entries = [e for e in entries if e.get("variant") == variant]
    if scheme:
        entries = [e for e in entries if e.get("scheme") == scheme]
    if role:
        entries = [e for e in entries if e.get("role") == role]
    if limit:
        entries = entries[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    resources = load_probe_resources(variant, device=device)
    resources["variant"] = {"tag": variant}

    for entry in entries:
        pid = int(entry["probe_id"])
        fname = (
            f"{entry.get('scheme','na')}_{entry.get('cluster','na')}_"
            f"{entry.get('role','canonical')}_pid{pid}.json"
        )
        path = out_dir / fname
        print(f"Generating graph for probe {pid} -> {path.name}")
        graph = generate_probe_graph(
            resources,
            bundle_index=int(entry["bundle_index"]),
            probe_id=pid,
            playbook_entry=entry,
        )
        path.write_text(json.dumps(graph, indent=2))
        if recommend:
            process_graph(path)


def regenerate_playbook_via_api(
    engine_url: str,
    playbook_path: Path,
    variant: str,
    out_dir: Path,
    *,
    scheme: str | None = None,
    limit: int | None = None,
    recommend: bool = True,
):
    entries = load_playbook_entries(playbook_path)
    entries = [e for e in entries if e.get("variant") == variant]
    if scheme:
        entries = [e for e in entries if e.get("scheme") == scheme]
    if limit:
        entries = entries[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    endpoint = f"{engine_url.rstrip('/')}/api/attribution/generate-graph"

    for entry in entries:
        pid = int(entry["probe_id"])
        fname = (
            f"{entry.get('scheme','na')}_{entry.get('cluster','na')}_"
            f"{entry.get('role','canonical')}_pid{pid}.json"
        )
        path = out_dir / fname
        payload = {"prompt": f"probe:{pid}"}
        print(f"POST probe:{pid}")
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        graph = response.json()
        graph.setdefault("metadata", {}).update(
            {
                k: entry[k]
                for k in ("scheme", "cluster", "role", "differential_top_k")
                if k in entry
            }
        )
        path.write_text(json.dumps(graph, indent=2))
        if recommend:
            process_graph(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate Neuronpedia graph JSONs")
    parser.add_argument(
        "--engine-url",
        type=str,
        default="http://localhost:8000",
        help="LeWM interpretability engine base URL (training mode)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under graphs/lewm-robot/",
    )
    parser.add_argument(
        "--playbook",
        type=str,
        default=None,
        help="neuronpedia_probe_playbook.json for static probes",
    )
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--scheme", type=str, default=None)
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run attribution in-process (no HTTP server)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output dir for playbook graphs",
    )
    parser.add_argument("--no-recommend", action="store_true")
    args = parser.parse_args()

    le_probe = Path(__file__).resolve().parents[2]

    if args.playbook:
        playbook_path = Path(args.playbook)
        if not args.variant:
            parser.error("--variant required with --playbook")
        out = Path(
            args.out_dir
            or le_probe
            / "workspace_visualization"
            / "attribution_graphs"
            / args.variant
        )
        if args.local:
            regenerate_playbook_local(
                playbook_path,
                args.variant,
                out,
                scheme=args.scheme,
                role=args.role,
                limit=args.limit,
                recommend=not args.no_recommend,
            )
        else:
            regenerate_playbook_via_api(
                args.engine_url,
                playbook_path,
                args.variant,
                out,
                scheme=args.scheme,
                limit=args.limit,
                recommend=not args.no_recommend,
            )
    else:
        regenerate_training_via_api(args.engine_url, args.output_subdir)
