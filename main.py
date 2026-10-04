import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from utils.config import settings
from utils.logger import logger
from api.middleware.cors import setup_cors
from api.routes import students, attendance, reports, alerts, auth, admin, camera, settings as policy_settings
from api.models.response import SystemStatusResponse
from ai_engine.camera.rtsp_client import camera_stream
from ai_engine.face_recognition.recognizer import face_recognizer
from database.queries import get_all_students, get_alerts_list
from database.queries import DatabaseUnavailableError
from utils.sections import resolve_allowed_sections
from utils.tokens import decode_token
from websocket_manager import manager, set_event_loop

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SmartCCTV AI Backend Service...")
    set_event_loop(asyncio.get_running_loop())
    # Optionally start camera ingestion thread
    try:
        camera_stream.start()
    except Exception as e:
        logger.warning(f"Could not start camera on boot: {e}")
    yield
    logger.info("Shutting down SmartCCTV AI Backend Service...")
    camera_stream.stop()

app = FastAPI(
    title="SmartCCTV API",
    description="AI-Powered Attendance and Campus Security System",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(_, exc: DatabaseUnavailableError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})

# Setup CORS
setup_cors(app)

# Register API Routers
app.include_router(students.router, prefix="/api")
app.include_router(attendance.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(camera.router, prefix="/api")
app.include_router(policy_settings.router, prefix="/api")

# System Status Endpoint
@app.get("/api/system/status", response_model=SystemStatusResponse)
def get_system_status():
    students_count = len(get_all_students())
    active_alerts = len(get_alerts_list(is_resolved=False))
    return SystemStatusResponse(
        status="online",
        camera_active=camera_stream.is_connected,
        ai_active=True,
        camera_index=settings.CAMERA_INDEX,
        fps=settings.CAMERA_FPS,
        recognition_model=face_recognizer.backend_name,
        active_students_count=students_count,
        unresolved_alerts_count=active_alerts,
        attendance_recording=camera_stream.attendance_recording,
    )

# WebSocket Endpoint for real-time live events
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Teachers pass their session token so check-in events stay inside their
    # assigned sections; other clients (camera kiosk) receive everything.
    claims = decode_token(websocket.query_params.get("token"))
    await manager.connect(websocket, resolve_allowed_sections(claims))
    try:
        # Send initial greeting
        await websocket.send_text(json.dumps({
            "type": "connection_ack",
            "message": "Connected to SmartCCTV Real-time Stream",
            "camera_index": settings.CAMERA_INDEX
        }))
        while True:
            data = await websocket.receive_text()
            # Echo or process incoming commands from frontend
            logger.debug(f"Received WS message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Lightweight liveness probe (no DB) used by Railway's healthcheck.
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "SmartCCTV AI Backend"}


def _frontend_dist() -> Path:
    dist = Path(settings.FRONTEND_DIST_PATH or "frontend/dist")
    if not dist.is_absolute():
        dist = Path(__file__).resolve().parent.parent / dist
    return dist


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    """Serve the built frontend; client-side routes fall back to index.html."""
    if full_path.startswith(("api/", "ws", "docs", "openapi.json", "redoc")):
        raise HTTPException(status_code=404)
    dist = _frontend_dist()
    candidate = (dist / full_path).resolve()
    if candidate.is_file() and candidate.is_relative_to(dist.resolve()):
        return FileResponse(candidate)
    index = dist / "index.html"
    if index.is_file():
        return FileResponse(index)
    raise HTTPException(status_code=404, detail="Frontend build not found; run `npm run build` in frontend/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
