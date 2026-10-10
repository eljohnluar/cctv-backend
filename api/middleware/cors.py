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
    allow_all = False

    for origin in extra_origins.split(","):
        origin = origin.strip().rstrip("/")
        if origin == "*":
            allow_all = True
        elif origin and origin not in origins:
            origins.append(origin)

    if allow_all:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_origin_regex=r"^https://.*(\.vercel\.app|\.onrender\.com)$",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
