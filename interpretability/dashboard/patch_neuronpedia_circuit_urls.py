#!/usr/bin/env python3
"""Add neuronpedia_url to .circuit.json files and refresh highlights doc pinned URLs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from extract_probe_circuits import (
    circuit_slug_from_path,
    neuronpedia_url_from_circuit,
)

NEURONPEDIA_URL_RE = re.compile(r"http://localhost:3000/lewm-robot/graph\?[^\s|)]+")

MASTER_TABLE_START = "<!-- MASTER_URL_TABLE_START -->"
MASTER_TABLE_END = "<!-- MASTER_URL_TABLE_END -->"

PLAYBOOK_VARIANTS = (
    "singleview",
    "multiview",
    "multiview_skeleton",
    "multiview_skeleton_dino",
)

SCHEME_ORDER = {"lateral": 0, "distance": 1, "pose": 2}
ROLE_ORDER = {"canonical": 0, "borderline": 1}


def playbook_slug(entry: dict) -> str:
    variant = entry["variant"]
    scheme = entry["scheme"]
    cluster = entry["cluster"]
    role = entry["role"]
    pid = entry["probe_id"]
    if scheme == "pose":
        return f"{variant}_{scheme}_{cluster}_{role}_pid{pid}"
    return f"{variant}_{scheme}_{cluster}_{role}_pid{pid}"


def sort_playbook_entries(entries: list[dict]) -> list[dict]:
    def key(entry: dict) -> tuple:
        return (
            SCHEME_ORDER.get(entry["scheme"], 99),
            entry["cluster"],
            ROLE_ORDER.get(entry["role"], 99),
            entry.get("rank_in_cluster", 0),
            (
                PLAYBOOK_VARIANTS.index(entry["variant"])
                if entry["variant"] in PLAYBOOK_VARIANTS
                else 99
            ),
            entry["probe_id"],
        )

    return sorted(entries, key=key)


def generate_master_table_md(
    playbook_path: Path,
    slug_to_url: dict[str, str],
) -> str:
    playbook = json.loads(playbook_path.read_text())
    entries = [e for e in playbook["entries"] if e["variant"] in PLAYBOOK_VARIANTS]
    entries = sort_playbook_entries(entries)

    lines = [
        "## All playbook graph URLs (132 rows: 33 probes × 4 variants)",
        "",
        "Pinned Neuronpedia links for every playbook graph. Each row is one `(variant, probe)`; "
        "`pinnedIds` come from `circuits/<slug>.circuit.json` (≤15 nodes). "
        "Regenerate with `patch_neuronpedia_circuit_urls.py --all-circuits`.",
        "",
        MASTER_TABLE_START,
        "",
        "| # | Variant | Scheme | Cluster | Role | Probe | Pinned URL |",
        "|---|---------|--------|---------|------|-------|------------|",
    ]

    missing = 0
    for idx, entry in enumerate(entries, start=1):
        slug = playbook_slug(entry)
        url = slug_to_url.get(slug, "")
        if not url:
            missing += 1
            url = f"*(missing circuit for `{slug}`)*"
        lines.append(
            f"| {idx} | {entry['variant']} | {entry['scheme']} | {entry['cluster']} | "
            f"{entry['role']} | {entry['probe_id']} | {url} |"
        )

    lines.extend(
        [
            "",
            MASTER_TABLE_END,
            "",
            f"**Coverage:** {len(entries) - missing}/{len(entries)} URLs from circuit metadata.",
            "",
            "| Variant tag | Slug prefix | Checkpoint dir | Seed command |",
            "|-------------|-------------|----------------|--------------|",
            "| `singleview` | `singleview_` | `checkpoints/lewm_grasp_baseline` | "
            "`seed_probe_graphs.js singleview` |",
            "| `multiview` | `multiview_` | `checkpoints/lewm_grasp_multiview` | "
            "`seed_probe_graphs.js multiview` |",
            "| `multiview_skeleton` | `multiview_skeleton_` | "
            "`checkpoints/lewm_grasp_multiview_skeleton` | "
            "`seed_probe_graphs.js multiview_skeleton` |",
            "| `multiview_skeleton_dino` | `multiview_skeleton_dino_` | "
            "`checkpoints/lewm_grasp_multiview_skeleton_dino` | "
            "`seed_probe_graphs.js multiview_skeleton_dino` |",
        ]
    )
    if missing:
        lines.append("")
        lines.append(f"**Warning:** {missing} slug(s) missing from circuits dir.")
    return "\n".join(lines) + "\n"


def replace_master_table_section(doc_path: Path, table_md: str) -> bool:
    text = doc_path.read_text()
    start = text.find(MASTER_TABLE_START)
    end = text.find(MASTER_TABLE_END)
    if start != -1 and end != -1:
        end = text.find("\n", end + len(MASTER_TABLE_END))
        if end == -1:
            end = len(text)
        new_text = (
            text[:start]
            + table_md.split(MASTER_TABLE_START, 1)[1].split(MASTER_TABLE_END, 1)[0]
            + text[end:]
        )
        # table_md includes markers; splice full section by title
        pattern = re.compile(
            r"## All playbook graph URLs \(132 rows:.*?\n"
            r"(?=---\n\n## Circuit extraction|\Z)",
            re.DOTALL,
        )
        if pattern.search(text):
            doc_path.write_text(pattern.sub(table_md + "\n---\n\n", text))
            return True
        doc_path.write_text(new_text)
        return True

    # First run: replace old MV+Skel-only index + Other variants block
    old_pattern = re.compile(
        r"## All `multiview_skeleton` canonical URLs \(pinned\).*?"
        r"(?=---\n\n## Circuit extraction)",
        re.DOTALL,
    )
    if old_pattern.search(text):
        doc_path.write_text(old_pattern.sub(table_md + "\n---\n\n", text))
        return True

    insert_before = "## Circuit extraction (backward paths)"
    if insert_before in text:
        doc_path.write_text(text.replace(insert_before, table_md + insert_before))
        return True
    return False


def patch_circuit_files(
    circuits_dir: Path, variant_prefix: str | None
) -> dict[str, str]:
    slug_to_url: dict[str, str] = {}
    for path in sorted(circuits_dir.glob("*.circuit.json")):
        if variant_prefix and not path.name.startswith(f"{variant_prefix}_"):
            if variant_prefix == "multiview" and path.name.startswith(
                "multiview_skeleton"
            ):
                continue
            elif variant_prefix == "multiview_skeleton" and path.name.startswith(
                "multiview_skeleton_dino_"
            ):
                continue
            elif not path.name.startswith(f"{variant_prefix}_"):
                continue
        circuit = json.loads(path.read_text())
        slug = circuit_slug_from_path(path)
        url = neuronpedia_url_from_circuit(circuit, path)
        extraction = circuit.setdefault("metadata", {}).setdefault(
            "circuit_extraction", {}
        )
        extraction["neuronpedia_url"] = url
        path.write_text(json.dumps(circuit, indent=2) + "\n")
        slug_to_url[slug] = url
    return slug_to_url


def refresh_highlights_doc(doc_path: Path, slug_to_url: dict[str, str]) -> int:
    text = doc_path.read_text()
    replaced = 0

    def sub(match: re.Match[str]) -> str:
        nonlocal replaced
        old = match.group(0)
        slug_m = re.search(r"slug=([^&]+)", old)
        if not slug_m:
            return old
        slug = slug_m.group(1)
        new = slug_to_url.get(slug)
        if new and new != old:
            replaced += 1
            return new
        return old

    new_text = NEURONPEDIA_URL_RE.sub(sub, text)
    doc_path.write_text(new_text)
    return replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuits-dir",
        type=Path,
        default=Path(
            "interpretability/neuronpedia/apps/webapp/public/graphs/lewm-robot/circuits"
        ),
    )
    parser.add_argument(
        "--highlights-doc",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "docs"
        / "reports"
        / "2026-05-29_neuronpedia_cluster_highlights.md",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="multiview_skeleton",
        help="Only refresh doc URLs for this variant prefix (default multiview_skeleton)",
    )
    parser.add_argument(
        "--all-circuits",
        action="store_true",
        help="Patch every .circuit.json in circuits-dir (not only --variant)",
    )
    parser.add_argument(
        "--playbook",
        type=Path,
        default=Path("workspace_visualization/neuronpedia_probe_playbook.json"),
        help="Playbook JSON for master URL table generation",
    )
    parser.add_argument(
        "--skip-master-table",
        action="store_true",
        help="Do not rewrite the all-variants master URL table section",
    )
    args = parser.parse_args()

    circuits_dir = args.circuits_dir.resolve()
    variant = None if args.all_circuits else args.variant
    slug_to_url = patch_circuit_files(circuits_dir, variant)
    doc_map = slug_to_url
    if args.variant and not args.all_circuits:
        prefix = f"{args.variant}_"
        doc_map = {
            k: v
            for k, v in slug_to_url.items()
            if k.startswith(prefix)
            and (
                args.variant != "multiview_skeleton"
                or not k.startswith("multiview_skeleton_dino_")
            )
        }

    highlights_doc = args.highlights_doc.resolve()
    n = refresh_highlights_doc(highlights_doc, doc_map)
    print(f"Patched {len(slug_to_url)} circuit file(s) with neuronpedia_url")
    print(
        f"Updated {n} URL(s) in {args.highlights_doc} ({len(doc_map)} slugs matched in doc)"
    )

    if not args.skip_master_table:
        playbook_path = args.playbook.resolve()
        table_md = generate_master_table_md(playbook_path, slug_to_url)
        if replace_master_table_section(highlights_doc, table_md):
            print(f"Rewrote master URL table in {args.highlights_doc}")
        else:
            print(f"Could not locate master table section in {args.highlights_doc}")


if __name__ == "__main__":
    main()
