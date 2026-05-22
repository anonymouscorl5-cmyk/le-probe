"""
HTTP transport for remote inference and teleop (msgpack bodies). Works with ``ngrok http <port>``.

- MPC / VLA: ``POST /plan``
- Teleop dashboard: ``POST /teleop``
"""

from __future__ import annotations

import msgpack
import numpy as np
import requests

DEFAULT_TIMEOUT_S = 120.0
TELEOP_TIMEOUT_S = 300.0
PLAN_PATH = "/plan"
TELEOP_PATH = "/teleop"
HEALTH_PATH = "/health"


def pack_np(arr: np.ndarray) -> dict:
    return {
        "data": arr.tobytes(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def unpack_np(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"])


def encode_body(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def decode_body(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)


class InferenceHTTPClient:
    """Synchronous msgpack-over-HTTP client."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        endpoint: str = PLAN_PATH,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.endpoint = endpoint
        self._headers = {
            "Content-Type": "application/msgpack",
            "Accept": "application/msgpack",
            "Ngrok-Skip-Browser-Warning": "true",
        }

    def health(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}{HEALTH_PATH}",
                headers=self._headers,
                timeout=10,
            )
            return r.status_code == 200
        except Exception:
            return False

    def post(self, payload: dict) -> dict:
        r = requests.post(
            f"{self.base_url}{self.endpoint}",
            data=encode_body(payload),
            headers=self._headers,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return decode_body(r.content)

    def plan(self, payload: dict) -> dict:
        """MPC / VLA planning request."""
        return self.post(payload)

    def command(self, payload: dict) -> dict:
        """Teleop command (reset, IK, joint targets, recording, etc.)."""
        return self.post(payload)


def create_app(
    handler, rpc_path: str = PLAN_PATH, title: str = "Cortex Inference Server"
):
    """Build a FastAPI app: ``handler(req_dict) -> resp_dict``."""
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI(title=title)

    @app.get(HEALTH_PATH)
    async def health():
        return {"status": "ok", "transport": "http+msgpack", "rpc_path": rpc_path}

    @app.post(rpc_path)
    async def rpc(request: Request):
        try:
            req = decode_body(await request.body())
            resp = handler(req)
            status = 500 if "error" in resp and "status" not in resp else 200
            return Response(
                content=encode_body(resp),
                media_type="application/msgpack",
                status_code=status,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            return Response(
                content=encode_body({"error": str(e)}),
                media_type="application/msgpack",
                status_code=500,
            )

    return app


def serve_http(
    handler,
    host: str = "0.0.0.0",
    port: int = 5555,
    rpc_path: str = PLAN_PATH,
    title: str = "Cortex Inference Server",
) -> None:
    import uvicorn

    app = create_app(handler, rpc_path=rpc_path, title=title)
    print(
        f"🌐 HTTP server: http://{host}:{port}  "
        f"(POST {rpc_path}, GET {HEALTH_PATH})"
    )
    print(f"   ngrok: ngrok http {port} --log=stdout")
    uvicorn.run(app, host=host, port=port, log_level="info")
