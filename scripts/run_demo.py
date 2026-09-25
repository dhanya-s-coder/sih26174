"""Run the existing HAR pipeline with its local OpenCV experiment view."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.har_space.api.runtime import HARApplication
from src.har_space.gui.overlay import draw_overlay


logger = logging.getLogger("har_demo")


def _status_messages(app: HARApplication) -> list[str]:
    messages = []
    if app.source.error_message:
        messages.append(f"Camera/video error: {app.source.error_message}")
    elif not app.source.running:
        messages.append("Camera/video source stopped or reached end of file.")

    if app.detector is None:
        messages.append(f"Color detection error: {app.detector_error}")
    if app.hand_landmarker is None:
        messages.append(f"Hand detection unavailable: {app.hand_error}")
    if app.voice.status == "ERROR":
        messages.append(f"Voice error: {app.voice.error_message}")
    if app.pipeline_error:
        messages.append(f"Pipeline error: {app.pipeline_error}")
    return messages


def _report_headless_errors(app: HARApplication, reported: set[str]) -> None:
    errors = []
    if app.source.error_message:
        errors.append(f"Camera/video error: {app.source.error_message}")
    if app.detector is None:
        errors.append(f"Color detection error: {app.detector_error}")
    if app.hand_landmarker is None:
        errors.append(f"Hand detection unavailable: {app.hand_error}")
    if app.voice.status == "ERROR":
        errors.append(f"Voice error: {app.voice.error_message}")
    if app.pipeline_error:
        errors.append(f"Pipeline error: {app.pipeline_error}")
    for error in errors:
        if error not in reported:
            logger.error(error)
            reported.add(error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SIH26174 HAR OpenCV demo")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--experiment", default="configs/experiment_demo.yaml")
    parser.add_argument("--mute", action="store_true", help="Disable speech output")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without an OpenCV window and exit when a file source ends",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = HARApplication(
        source_arg=args.source,
        experiment_path=args.experiment,
        mute=args.mute,
        save_annotated=args.save_annotated,
    )
    window_name = "SIH26174 - HAR Experiment"
    reported_errors: set[str] = set()

    try:
        app.start()
        if not args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            if args.headless:
                _report_headless_errors(app, reported_errors)
            snapshot = app.runtime.snapshot()
            image = snapshot.frame.copy() if snapshot.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            instruction = (
                app.spec.steps[snapshot.step_index].instruction
                if snapshot.step_index < len(app.spec.steps)
                else "Experiment complete."
            )
            view = draw_overlay(
                image,
                list(snapshot.detections),
                list(snapshot.hands),
                app.spec,
                snapshot.step_index,
                set(snapshot.completed_step_ids),
                active_alert=snapshot.latest_alert,
                show_telemetry=True,
                interaction_text=snapshot.current_interaction,
                current_instruction=instruction,
                status_messages=_status_messages(app),
                step_statuses=snapshot.step_statuses,
            )

            if args.headless:
                pipeline_thread = app.pipeline_thread
                if not app.source.running and (pipeline_thread is None or not pipeline_thread.is_alive()):
                    break
                time.sleep(0.02)
                continue

            cv2.imshow(window_name, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                if not app.reset_experiment():
                    logger.warning("Reset was not applied; the frame-processing pipeline is not running.")
    except KeyboardInterrupt:
        logger.info("Stopping the HAR demo.")
    except Exception:
        logger.exception("HAR demo failed")
        raise
    finally:
        app.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()