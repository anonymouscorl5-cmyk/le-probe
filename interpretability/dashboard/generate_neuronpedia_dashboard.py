import json
import argparse
from pathlib import Path


def generate_dashboard(
    report_path, output_path, model_name="LeWM-v17", layer="encoder_L0"
):
    """
    Converts a Le-Probe feature audit report into a Neuronpedia-compatible JSON dashboard.
    """
    print(f"🎨 Generating Neuronpedia Dashboard from {report_path}...")

    with open(report_path, "r") as f:
        report = json.load(f)

    dashboard = {"model_name": model_name, "layer": layer, "features": []}

    for fid, examples in report.items():
        # Neuronpedia expects a list of top activations for each feature
        feature_data = {"feature_index": int(fid), "activations": []}

        for val, idx in examples:
            # We use the Semantic URI Pattern for visual tokens
            token_str = f"<|IMG_{idx}|>"
            feature_data["activations"].append(
                {
                    "token": token_str,
                    "value": round(val, 4),
                    "context": [token_str],  # Visual features use single-token context
                }
            )

        dashboard["features"].append(feature_data)

    with open(output_path, "w") as f:
        json.dump(dashboard, f, indent=4)

    print(f"✨ Dashboard generated: {output_path}")
    print(
        f"💡 You can now upload this file to Neuronpedia or use it with a local patched instance."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=str, required=True, help="Path to feature_audit_report.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="neuronpedia_dashboard.json",
        help="Output filename",
    )
    parser.add_argument(
        "--model", type=str, default="LeWM-v17", help="Model name for metadata"
    )
    parser.add_argument(
        "--layer", type=str, default="encoder_L0", help="Layer ID for metadata"
    )
    args = parser.parse_args()

    generate_dashboard(args.report, args.output, args.model, args.layer)
