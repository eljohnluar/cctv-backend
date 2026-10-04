"""WebSocket event delivery shared by API routes and camera worker threads."""

import asyncio
import json
from typing import Dict, List, Optional

from fastapi import WebSocket

from utils.logger import logger


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        # Per-connection section scope; None means the client sees every section.
        self.scopes: Dict[WebSocket, Optional[List[str]]] = {}

    async def connect(self, websocket: WebSocket, allowed_sections: Optional[List[str]] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.scopes[websocket] = allowed_sections
        logger.info("WebSocket client connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        self.scopes.pop(websocket, None)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected. Total clients: %d", len(self.active_connections))

    def _visible(self, websocket: WebSocket, message: dict) -> bool:
        if message.get("type") not in ("attendance", "attendance_time_out"):
            return True
        allowed = self.scopes.get(websocket)
        return allowed is None or message.get("section") in allowed

    async def broadcast(self, message: dict):
        for connection in self.active_connections.copy():
            if not self._visible(connection, message):
                continue
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()
_event_loop: Optional[asyncio.AbstractEventLoop] = None


def set_event_loop(event_loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Register the application loop so camera worker threads can publish safely."""
    global _event_loop
    _event_loop = event_loop


def publish_from_worker(message: dict) -> None:
    """Queue a WebSocket message from synchronous camera-recognition work."""
    if _event_loop is None or not _event_loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(message), _event_loop)
