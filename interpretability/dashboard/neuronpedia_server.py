import os
import argparse
import requests
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
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
}


@app.get("/api/robot-dataset/frames/{idx}.jpg")
async def get_frame(idx: int):
    """
    Proxies a frame request to the Colab/Pinggy bridge.
    The bridge handles all the math for mapping the index to a dataset sample and patch.
    """
    if not CONFIG["remote_url"]:
        raise HTTPException(
            status_code=501,
            detail="Cloud bridge URL not configured. Set COLAB_BRIDGE_URL in .env or use --remote-url",
        )

    # REMOTE MODE: Proxy to Colab/Pinggy
    url = f"{CONFIG['remote_url'].rstrip('/')}/api/robot-dataset/frames/{idx}.jpg"
    try:
        # Increase timeout for cloud bridge (some frames might take a second to extract)
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            error_detail = f"Remote bridge error {resp.status_code}: {resp.text[:200]}"
            print(f"❌ {error_detail}")
            raise HTTPException(status_code=resp.status_code, detail=error_detail)

        return Response(content=resp.content, media_type="image/jpeg")

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Connection to Colab bridge failed. Is the Pinggy tunnel alive? {e}",
        )
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "remote_url": CONFIG["remote_url"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuronpedia Visual Proxy")
    parser.add_argument(
        "--remote-url",
        type=str,
        help="Pinggy/Ngrok URL of the Colab bridge (overrides .env)",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Priority: 1. CLI Arg (--remote-url) | 2. Env Var (COLAB_BRIDGE_URL)
    CONFIG["remote_url"] = args.remote_url or os.getenv("COLAB_BRIDGE_URL")

    import uvicorn

    print(f"🚀 Neuronpedia Visual Proxy starting on port {args.port}")
    if CONFIG["remote_url"]:
        print(f"🔗 Proxying to cloud: {CONFIG['remote_url']}")
    else:
        print(
            "⚠️  Warning: No remote URL configured. Requests will fail until COLAB_BRIDGE_URL is set."
        )

    uvicorn.run(app, host="0.0.0.0", port=args.port)
