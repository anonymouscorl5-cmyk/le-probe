import os
import json
import torch
import argparse
from pathlib import Path


def export_to_saelens(input_path, output_dir):
    """
    Converts Le-Probe Transcoder/SAE weights to the SAELens/Neuronpedia format.
    """
    print(f"📦 Exporting {input_path} to SAELens format...")

    # 1. Load Le-Probe weights
    data = torch.load(input_path, map_location="cpu")
    sd = data["state_dict"]
    config = data["config"]
    norm_stats = data["norm_stats"]

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # 2. Map Weights
    # SAELens expects:
    # W_enc: (d_model, d_sae)
    # b_enc: (d_sae,)
    # W_dec: (d_sae, d_model)
    # b_dec: (d_model,)

    new_sd = {
        "W_enc": sd["encoder.weight"].T,
        "b_enc": sd["encoder.bias"],
        "W_dec": sd["decoder.weight"].T,
        "b_dec": sd["decoder.bias"],
    }

    # 3. Generate SAELens Config
    saelens_config = {
        "architecture": (
            "standard"
            if config["source_layer"] == config["target_layer"]
            else "transcoder"
        ),
        "d_in": config.get("d_model", 192),
        "d_sae": config["dict_size"],
        "dtype": "float32",
        "device": "cpu",
        "model_name": "LeWM-v17",
        "hook_name": config["source_layer"],
        "hook_layer": (
            int(config["source_layer"].split("_L")[-1])
            if "_L" in config["source_layer"]
            else 0
        ),
        "activation_size": config.get("d_model", 192),
        "norm_stats": {
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
        },
    }

    # 4. Save
    torch.save(new_sd, output_path / "sae_weights.pt")
    with open(output_path / "config.json", "w") as f:
        json.dump(saelens_config, f, indent=4)

    print(f"✨ Export complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path to .pt weight file"
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    export_to_saelens(args.input, args.output)
