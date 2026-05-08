import os
import json
import requests
import sys


def regenerate_via_api(engine_url):
    print(f"🚀 Using Engine API at: {engine_url}")

    scenarios = [
        {"slug": "grasp-success-1", "index": 452, "joint": 7},
        {"slug": "grasp-fail-1", "index": 120, "joint": 7},
        {"slug": "approach-can", "index": 800, "joint": 7},
        {"slug": "pre-grasp-pos", "index": 300, "joint": 7},
    ]

    # Output directory
    out_dir = (
        f"{os.getcwd()}/interpretability/neuronpedia/apps"
        "/webapp/public/graphs/lewm-robot"
    )
    os.makedirs(out_dir, exist_ok=True)

    for scene in scenarios:
        print(
            f"📊 Fetching attribution for {scene['slug']} (index {scene['index']})..."
        )

        endpoint = f"{engine_url}/api/attribution/generate-graph"
        payload = {"prompt": f"{scene['index']}:{scene['joint']}"}

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()

            graph_data = response.json()

            file_path = os.path.join(out_dir, f"{scene['slug']}.json")
            with open(file_path, "w") as f:
                json.dump(graph_data, f, indent=2)
            print(f"✅ Saved to {file_path}")

        except Exception as e:
            print(f"❌ Failed to fetch {scene['slug']}: {e}")


if __name__ == "__main__":
    url = "http://localhost:8080"  # Default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    regenerate_via_api(url)
