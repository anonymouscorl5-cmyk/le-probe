import os
import json
import argparse
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

app = FastAPI()

# Allow CORS for local Neuronpedia instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
CONFIG = {
    "remote_url": None,
    "dashboard_path": None,
}


@app.get("/dashboard")
async def get_dashboard():
    """Returns the Neuronpedia-compatible dashboard JSON."""
    if not CONFIG["dashboard_path"].exists():
        raise HTTPException(status_code=404, detail="Dashboard JSON not found")
    with open(CONFIG["dashboard_path"], "r") as f:
        return json.load(f)


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Proxies or locally extracts a frame based on activation index.
    """
    if CONFIG["remote_url"]:
        # REMOTE MODE: Proxy to Colab/Pinggy
        url = f"{CONFIG['remote_url'].rstrip('/')}/api/robot-dataset/frames/{idx}.jpg"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                error_detail = f"Remote error {resp.status_code}: {resp.text[:200]}"
                print(f"❌ {error_detail}")
                raise HTTPException(status_code=resp.status_code, detail=error_detail)
            return Response(content=resp.content, media_type="image/jpeg")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            raise HTTPException(
                status_code=502, detail=f"Connection to Colab failed: {e}"
            )
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # LOCAL MODE: Reverting to original CV2 logic (for small local subsets)
        # Note: This requires the local paths to be configured correctly
        raise HTTPException(
            status_code=501, detail="Local extraction not configured in proxy mode"
        )


if __name__ == "__main__":
    # Priority: 1. CLI Arg (--remote-url) | 2. Env Var (COLAB_BRIDGE_URL)
    COLAB_BRIDGE_URL = os.getenv("COLAB_BRIDGE_URL")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dashboard", type=str, required=True, help="Path to dashboard JSON"
    )
    parser.add_argument(
        "--remote-url", type=str, help="Pinggy/Ngrok URL of the Colab bridge"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    CONFIG["dashboard_path"] = Path(args.dashboard)
    CONFIG["remote_url"] = args.remote_url or COLAB_BRIDGE_URL

    import uvicorn

    print(f"🚀 Neuronpedia Visual Bridge starting on port {args.port}")
    if CONFIG["remote_url"]:
        print(f"🔗 Proxying to cloud: {CONFIG['remote_url']}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
