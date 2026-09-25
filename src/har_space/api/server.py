"""Local FastAPI bridge for the existing HAR runtime."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.har_space.api.runtime import HARApplication


def _snapshot_payload(app: HARApplication) -> dict[str, Any]:
    snapshot = app.runtime.snapshot()
    steps = []
    for index, step in enumerate(app.spec.steps):
        status = snapshot.step_statuses.get(step.id)
        if status not in {"completed", "skipped", "out_of_sequence", "failed"}:
            status = "current" if index == snapshot.step_index and snapshot.experiment_status != "COMPLETE" else "pending"
        steps.append({
            "id": step.id,
            "position": step.position,
            "name": step.name,
            "instruction": step.instruction,
            "status": status,
            "outcome": step.expected_outcome,
        })

    latest_alert = None
    if snapshot.latest_alert:
        latest_alert = {
            "message": snapshot.latest_alert.message,
            "level": snapshot.latest_alert.level,
            "timestamp": snapshot.latest_alert.timestamp,
            "expected": snapshot.alert_expected,
            "detected": snapshot.alert_detected,
        }

    current_step = steps[snapshot.step_index] if snapshot.step_index < len(steps) else None
    activity = [
        {
            "time": datetime.fromtimestamp(item.received_at, timezone.utc).isoformat(),
            "event": item.description,
            "category": item.category,
            "status": item.level,
        }
        for item in snapshot.recent_activity
    ]

    return {
        "system": "error" if snapshot.experiment_status == "ERROR" else snapshot.experiment_status.lower(),
        "experiment": {"name": app.spec.name, "version": app.spec.version},
        "progress": {
            "current_index": snapshot.step_index,
            "total": len(steps),
            "current_step": current_step,
            "completed": [step for step in steps if step["status"] == "completed"],
            "steps": steps,
            "complete": snapshot.experiment_status == "COMPLETE",
            "duration_seconds": snapshot.duration_seconds if snapshot.experiment_status == "COMPLETE" else None,
            "error_count": snapshot.error_count,
        },
        "action": snapshot.current_interaction,
        "assistant": {"message": snapshot.latest_speech, "status": snapshot.health.get("tts", ("N/A", ""))[0]},
        "alert": latest_alert,
        "health": {
            "camera": snapshot.health.get("camera", ("N/A", ""))[0],
            "detection": snapshot.health.get("detector", ("N/A", ""))[0],
            "hand_tracking": snapshot.health.get("hands", ("N/A", ""))[0],
            "experiment_tracking": snapshot.health.get("tracker", ("N/A", ""))[0],
            "voice_assistant": snapshot.health.get("tts", ("N/A", ""))[0],
        },
        "camera": {"stream_url": f"http://127.0.0.1:{app.streamer.port}/", "frame_id": snapshot.frame_id},
        "activity": activity,
        "updated_at": datetime.fromtimestamp(snapshot.updated_at, timezone.utc).isoformat(),
    }


def create_app(runtime: HARApplication) -> FastAPI:
    """Build the HTTP/WebSocket interface around one existing HARApplication."""
    api = FastAPI(title="SIH26174 HAR Local API", version="1.0.0")
    api.state.har_runtime = runtime
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @api.get("/api/status")
    def status():
        payload = _snapshot_payload(runtime)
        return {"system": payload["system"], "health": payload["health"], "updated_at": payload["updated_at"]}

    @api.get("/api/experiment")
    def experiment():
        payload = _snapshot_payload(runtime)
        return payload["experiment"] | {"steps": payload["progress"]["steps"]}

    @api.get("/api/progress")
    def progress():
        return _snapshot_payload(runtime)["progress"]

    @api.post("/api/experiment/reset")
    async def reset_experiment():
        applied = await asyncio.to_thread(runtime.reset_experiment)
        if not applied:
            raise HTTPException(status_code=503, detail="HAR pipeline is not running or reset was not applied")
        return _snapshot_payload(runtime)

    @api.get("/api/logs")
    def logs(limit: int = 100):
        payload = _snapshot_payload(runtime)
        step_records = [
            {
                "time": record.iso_timestamp,
                "event": f"{record.step_name} — {record.status.value.upper()}: {record.outcome_text}",
                "category": "STEP RECORD",
                "status": "warning" if record.status.value in {"skipped", "out_of_sequence", "failed"} else "success",
            }
            for record in runtime.log_writer.records
        ]
        return {
            "items": payload["activity"][:max(1, min(limit, 500))],
            "step_records": step_records,
            "path": str(runtime.log_path),
        }

    @api.get("/api/state")
    def state():
        return _snapshot_payload(runtime)

    @api.get("/api/health")
    def health():
        return {"status": "ok", "system": _snapshot_payload(runtime)["system"]}

    @api.websocket("/ws")
    async def websocket_updates(websocket: WebSocket):
        await websocket.accept()
        last_payload = None
        try:
            while True:
                payload = _snapshot_payload(runtime)
                if payload != last_payload:
                    await websocket.send_json({"type": "runtime_update", "data": payload})
                    last_payload = payload
                await asyncio.sleep(0.25)
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return

    return api


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the existing HAR runtime with its local web API")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--experiment", default="configs/experiment_demo.yaml")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stream-port", type=int, default=8081)
    args = parser.parse_args()

    runtime = HARApplication(
        source_arg=args.source,
        experiment_path=args.experiment,
        mute=args.mute,
        save_annotated=args.save_annotated,
        stream_host="127.0.0.1",
        stream_port=args.stream_port,
    )
    runtime.start()
    try:
        uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="info")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
