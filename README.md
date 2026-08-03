# Covert Chess — BAM Demo

An interactive demonstration of **BAM** (Burnashev Adaptive Posterior Matching), a
feedback-coding scheme for inference-time covert communication between language
model agents. You play chess against a Stockfish engine, but your moves are
never transmitted in the clear: each move is embedded into ordinary,
natural-language chat text passed between two LLM agents. The receiving agent
recovers the move without it ever appearing explicitly in the conversation.

This repository contains two parts:

| Part | Path | Hosting | Purpose |
|------|------|---------|---------|
| **Static site** | `demo/` | GitHub Pages | Project landing page and write-up |
| **Live demo** | `backend/` | RunPod (GPU) | The playable, model-backed application |

- Project site: <https://sidongg.github.io/Covert_Chess/>
- Live demo: served from the RunPod pod (see **Deployment**)

---

## How it works

Each turn runs one covert exchange in each direction:

1. **You** make a chess move and type a short message.
2. **Agent A** (LLM 1) paraphrases your message into a longer, natural-sounding
   piece of cover text. Your move is embedded into that text via the BAM
   encoder — the text reads as ordinary conversation but secretly carries the
   move.
3. **Agent B** (LLM 2) receives only that text. It decodes your move using the
   shared BAM channel, queries Stockfish for a reply, and embeds the engine's
   move into its own cover-text response.
4. The board updates. Click **Decode** on either message to reveal the hidden
   move.

Agent B never sees the move written out — only the watermarked text. Both agents
run the same open model (`meta-llama/Llama-3.1-8B-Instruct`), and the covert
channel is realized with the ArcMark adapter over an entropy-coupled token
distribution.

> **Note on the shared board.** For clarity, the demo maintains a single
> authoritative chess board and always applies your true move. If the decoder
> ever recovers the wrong move, the UI flags it explicitly: in a real two-agent
> deployment the agents' boards would desync, but here the game continues on
> your real move. Nothing breaks.

---

## Repository layout

```
.
├── demo/
│   └── index.html          # Static GitHub Pages site
└── backend/
    ├── server.py           # FastAPI + WebSocket server
    ├── session.py          # DemoSession — one game, streaming inference
    ├── chess_engine.py     # Stockfish wrapper (ChessInterface)
    ├── lm_backend_shared.py# SharedModelPool (single GPU-resident model)
    ├── bam/                # BAM encoder / decoder / tracker / ArcMark adapter
    └── static/
        └── index.html      # Live-demo front end (served by server.py)
```

The front ends in `demo/index.html` and `backend/static/index.html` are kept in
sync; the static copy points users to the live pod, and the live copy links back
to the project site.

---

## Running the backend

**Requirements**

- Python 3.10+
- A CUDA GPU with sufficient VRAM for Llama-3.1-8B-Instruct
- [Stockfish](https://stockfishchess.org/) installed on the host
- A Hugging Face token with access to the Llama-3.1 weights

**Install**

```bash
cd backend
pip install -r requirements.txt   # fastapi, uvicorn, torch, transformers, python-chess, …
apt-get install -y stockfish      # or provide STOCKFISH_PATH
```

**Configure** (environment variables, all optional except `HF_TOKEN`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token for gated Llama weights |
| `MODEL_NAME` | `meta-llama/Llama-3.1-8B-Instruct` | Model both agents use |
| `STOCKFISH_PATH` | `/usr/games/stockfish` | Path to the Stockfish binary |
| `SKILL_LEVEL` | `5` | Default engine strength |
| `MAX_SESSIONS` | `20` | Concurrent session cap |

**Run**

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

`--workers 1` is required: the model is loaded once and shared across sessions
within a single process. GPU generation is serialized by an async lock so
concurrent players cannot collide.

Open `http://<host>:8000/` to play. A health probe is available at `/health`.

---

## Deployment

**Live demo (RunPod).** Deploy `backend/` to a GPU pod, run the uvicorn command
above, and expose port 8000 via the pod's HTTP proxy. The front end connects
over WebSocket; on GitHub Pages it uses the hard-coded pod URL in
`static/index.html`, or a `?backend=wss://…` query parameter, or a URL pasted
into the connect panel.

Because the page is served from disk, HTML changes take effect after a
`git pull` on the pod plus a browser hard-refresh. **Changes to Python
(`server.py`, `session.py`, `bam/`) require restarting uvicorn.**

**Static site (GitHub Pages).** `demo/index.html` is published from the repo.
Push to the tracked branch and GitHub Pages redeploys automatically.

---

## Configuration in the app

- **Engine strength** — Easy / Medium / Hard / Maximum, set from the board panel.
- **Agent prompts** — the system prompts for Agent A (paraphrase/relay) and
  Agent B (conversational reply) are editable at runtime from the settings
  panel and applied per session.

---

## Client ⇄ server protocol

A single WebSocket at `/ws/{session_id}`. The client sends `user_turn`,
`reset`, `set_prompts`, `get_prompts`, and `set_difficulty`. The server streams
back `ready`, `status`, `token` (per-token, per-agent), `turn_done`,
`decode_result`, `engine_move`, `board_update`, `game_over`, and prompt/config
acknowledgements. Sessions are keyed per browser tab and released on disconnect.

---

## Limitations

- The demo uses an 8B open model; it can occasionally produce repetitive or
  imperfect cover text.
- The covert channel is reliable but not infallible — decode mismatches are
  surfaced in the UI rather than hidden.
- One shared board is used for clarity; the true multi-agent desync scenario is
  described, not simulated.

---

## Citation

If you reference this work, please cite the accompanying paper,
*Feedback Coding Enables Inference-Time Covert Agentic Communication*.
