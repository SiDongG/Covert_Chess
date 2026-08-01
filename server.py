"""FastAPI WebSocket server — BAM chess demo.

Run (from backend/ directory):
    uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1

workers MUST be 1: the shared model lives in a single process.
"""
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lm_backend_shared import SharedModelPool
from bam.bam_tracker    import BAMConfig           # ← bam package
from bam.arcmark_adapter import ArcMarkConfig
from session             import DemoSession

MODEL_NAME     = os.getenv("MODEL_NAME",     "meta-llama/Llama-3.1-8B-Instruct")
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", "/usr/games/stockfish")
SKILL_LEVEL    = int(os.getenv("SKILL_LEVEL",  "5"))
MAX_SESSIONS   = int(os.getenv("MAX_SESSIONS", "4"))

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="BAM Chess Demo")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

pool: SharedModelPool | None = None
sessions: dict[str, DemoSession] = {}


@app.on_event("startup")
async def startup() -> None:
    global pool
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
    pool = SharedModelPool(MODEL_NAME)
    print("[server] Ready.")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions),
            "vram_gb": pool.vram_gb() if pool else 0}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            await websocket.send_json(
                {"type": "error", "msg": "Server at capacity — try again later."}
            )
            await websocket.close()
            return
        sessions[session_id] = DemoSession(
            lm1=pool.make_backend(),
            lm2=pool.make_backend(),
            stockfish_path=STOCKFISH_PATH,
            bam_cfg=BAMConfig(
                gamma_1=0.85, gamma_2=0.99,
                rho_ack=0.95, rho_nack=0.95,
                p_field=4, eps_noise=0.5,
            ),
            adapter_cfg=ArcMarkConfig(
                p_field=4, r_resolution=8,
                shared_seed=0xA12C, top_k=50,
                sinkhorn_max_iter=1000,
                sinkhorn_stop_thr=1e-4,
                sinkhorn_reg=0.2,
                sinkhorn_method="sinkhorn_log",
            ),
        )

    session = sessions[session_id]
    await websocket.send_json({
        "type": "ready", "session_id": session_id,
        "fen": session.chess.fen(), "turn": session.turn_count,
    })

    async def send(msg: dict) -> None:
        await websocket.send_json(msg)

    try:
        while True:
            data = await websocket.receive_json()
            t    = data.get("type")
            if t == "user_turn":
                await session.handle_user_turn(
                    chat=str(data.get("chat", "")),
                    move_uci=str(data.get("move", "")),
                    send=send,
                )
            elif t == "reset":
                session.reset()
                await send({"type": "reset_ok", "fen": "start", "turn": 0})
            elif t == "ping":
                await send({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"type": "error", "msg": str(exc)})
        except Exception:
            pass
