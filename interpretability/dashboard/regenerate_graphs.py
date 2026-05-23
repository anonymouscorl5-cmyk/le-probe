import os
import json
import argparse
import requests


def regenerate_via_api(engine_url: str, output_subdir: str = ""):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-generate Neuronpedia graph JSONs")
    parser.add_argument(
        "--engine-url",
        type=str,
        default="http://localhost:8000",
        help="LeWM interpretability engine base URL",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under graphs/lewm-robot/",
    )
    args = parser.parse_args()
    regenerate_via_api(args.engine_url, args.output_subdir)
