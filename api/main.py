"""
api/main.py — FastAPI application entry-point for RainSight India
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api.routes import predict as predict_router
from api.routes import train   as train_router
from api.routes import data    as data_router


# ─────────────────────────────────────────────────────────────────
# Lifespan: warm-up the predictor on startup (if model exists)
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[·] RainSight API starting up...")
    # Try to pre-load the predictor so first request is fast
    try:
        import predict as pm
        pm.get_predictor()
        print("[✓] Model loaded and ready.")
    except FileNotFoundError:
        print("[!] No trained model found. Train via POST /api/train.")
    except Exception as e:
        print(f"[!] Model load warning: {e}")
    yield
    print("[·] RainSight API shut down.")


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "RainSight India API",
    description = "CNN+LSTM hybrid rainfall forecasting for Indian meteorological subdivisions",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── API routes ────────────────────────────────────────────────────
app.include_router(predict_router.router, prefix="/api", tags=["Forecast"])
app.include_router(train_router.router,   prefix="/api", tags=["Training"])
app.include_router(data_router.router,    prefix="/api", tags=["Data"])


# ── Serve built frontend (if exists) ─────────────────────────────
FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Frontend not built yet. Run: cd frontend && npm run build"}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    "RainSight India",
        "version": "1.0.0",
        "docs":    "/docs",
        "api":     "/api",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
