from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from utils.config import settings


def setup_cors(app: FastAPI):
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "https://smartcctvnewapp.vercel.app",
    ]
    extra_origins = settings.CORS_ORIGINS or ""
    for origin in extra_origins.split(","):
        origin = origin.strip().rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
