"""
Load LeWM + workspace probes and run LeWMAttributor without the HTTP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml

from interpretability.dashboard.engine import LeWMAttributor
from interpretability.dashboard.workspace_probe_dataset import WorkspaceProbeDataset
from interpretability.lewm_experiment import (
    ExperimentConfig,
    build_goal_mapper_for_probes,
)

LE_PROBE_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = Path(__file__).parent / "variant_profiles.yaml"


def load_profiles(path: Path | None = None) -> dict:
    p = path or PROFILES_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str, profiles: dict) -> Path:
    root = profiles.get("repo_root", "..")
    base = (Path(__file__).parent / root).resolve()
    return (base / rel).resolve()


def variant_config(profiles: dict, variant_tag: str) -> dict:
    variants = profiles["variants"]
    if variant_tag not in variants:
        raise KeyError(
            f"Unknown variant '{variant_tag}'. Choose from: {list(variants)}"
        )
    return variants[variant_tag]


def experiment_config_from_variant(v: dict, profiles: dict) -> ExperimentConfig:
    d = profiles.get("defaults", {})
    num_views = 5 if v.get("multi_view") else 1
    return ExperimentConfig(
        multi_view=bool(v.get("multi_view")),
        use_skeleton=bool(v.get("use_skeleton")),
        use_dino=bool(v.get("use_dino")),
        num_views=num_views,
        history_size=int(d.get("history_size", 3)),
        attribution_target=str(v.get("attribution_target", "reward")),
    )


def load_pose_labels(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return {
        int(pid): str(lab) for pid, lab in zip(doc["probe_ids"], doc["segment_hint"])
    }


def load_probe_resources(
    variant_tag: str,
    *,
    profiles_path: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    profiles = load_profiles(profiles_path)
    v = variant_config(profiles, variant_tag)
    cfg = experiment_config_from_variant(v, profiles)
    defaults = profiles["defaults"]

    bundle_path = resolve_path(defaults["probe_bundle"], profiles)
    pose_path = resolve_path(defaults["pose_clusters"], profiles)
    ckpt_dir = resolve_path(v["checkpoint"], profiles)
    model_ckpt = ckpt_dir / "gr1_reward_tuned_v2.ckpt"
    if not model_ckpt.exists():
        candidates = list(ckpt_dir.glob("*.ckpt"))
        model_ckpt = candidates[0] if candidates else model_ckpt

    transcoder_dir = ckpt_dir / defaults.get(
        "transcoder_subdir", "transcoder_weights_residual"
    )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mapper = build_goal_mapper_for_probes(str(model_ckpt), cfg)
    model = mapper.model.to(dev).eval()

    transcoders: dict[str, dict] = {}
    for path in sorted(transcoder_dir.glob("*.pt")):
        layer_id = path.stem.split("_clt")[0].split("_sae")[0].split("_residual")[0]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        from interpretability.transcoders.universal_transcoder import Transcoder

        state_dict = checkpoint["state_dict"]
        norm_stats = checkpoint.get("norm_stats", {})
        d_dict, d_model = state_dict["encoder.weight"].shape
        d_output = state_dict["decoder.weight"].shape[0]
        tc = Transcoder(d_model, d_dict, d_output=d_output).to(dev)
        tc.load_state_dict(state_dict)
        transcoders[layer_id] = {"model": tc.eval(), "stats": norm_stats}

    dataset = WorkspaceProbeDataset(
        bundle_path, cfg, pose_labels=load_pose_labels(pose_path)
    )

    return {
        "profiles": profiles,
        "variant": v,
        "cfg": cfg,
        "device": dev,
        "model": model,
        "mapper": mapper,
        "transcoders": transcoders,
        "dataset": dataset,
        "min_k": int(defaults.get("min_k", 15)),
        "ig_steps": int(defaults.get("ig_steps", 20)),
        "bundle_path": bundle_path,
    }


def parse_probe_prompt(prompt: str) -> tuple[str, int]:
    """
    Returns ('probe', probe_id) or ('index', bundle_index).
    Formats: probe:127, pid:127, 127, index:84
    """
    clean = str(prompt).replace("<bos>", "").strip()
    if ":" in clean:
        kind, val = clean.split(":", 1)
        kind = kind.lower()
        if kind in ("probe", "pid"):
            return "probe", int(val)
        if kind in ("index", "idx"):
            return "index", int(val)
    if clean.isdigit():
        return "probe", int(clean)
    raise ValueError(f"Cannot parse probe prompt: {prompt!r}")


def resolve_bundle_index(dataset: WorkspaceProbeDataset, prompt: str) -> int:
    kind, val = parse_probe_prompt(prompt)
    if kind == "probe":
        return dataset.index_for_probe_id(val)
    return int(val)


def enrich_graph_metadata(
    graph: dict,
    *,
    bundle_index: int,
    probe_id: int,
    variant_tag: str,
    playbook_entry: dict | None = None,
) -> dict:
    meta = graph.setdefault("metadata", {})
    meta["bundle_index"] = bundle_index
    meta["probe_id"] = probe_id
    meta["variant"] = variant_tag
    if playbook_entry:
        for key in (
            "scheme",
            "cluster",
            "role",
            "differential_top_k",
            "overlap_with_cluster_top_k",
        ):
            if key in playbook_entry:
                meta[key] = playbook_entry[key]
    return graph


def generate_probe_graph(
    resources: dict[str, Any],
    *,
    bundle_index: int | None = None,
    probe_id: int | None = None,
    target_logit_idx: int = 7,
    playbook_entry: dict | None = None,
) -> dict:
    dataset: WorkspaceProbeDataset = resources["dataset"]
    if bundle_index is None:
        if probe_id is None:
            raise ValueError("Need bundle_index or probe_id")
        bundle_index = dataset.index_for_probe_id(probe_id)
    if probe_id is None:
        probe_id = dataset.probe_id_at(bundle_index)

    sample = dataset[bundle_index]
    batch = {
        k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
        if isinstance(v, torch.Tensor)
    }

    attributor = LeWMAttributor(
        resources["model"],
        resources["transcoders"],
        resources["mapper"],
        resources["cfg"],
        min_k=resources["min_k"],
    )
    graph = attributor.attribute(
        batch,
        target_logit_idx,
        steps=resources["ig_steps"],
        trace_id=bundle_index,
    )
    slug = f"probe_{probe_id}"
    graph["metadata"]["slug"] = slug
    graph["metadata"]["trace_id"] = bundle_index
    graph["metadata"]["prompt"] = f"probe:{probe_id}"
    graph["metadata"]["title_prefix"] = f"Probe {probe_id}"
    enrich_graph_metadata(
        graph,
        bundle_index=bundle_index,
        probe_id=probe_id,
        variant_tag=resources["variant"]["tag"],
        playbook_entry=playbook_entry,
    )
    return graph
