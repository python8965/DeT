from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from .detector import DetectorConfig, RedFlashDetector
from .notifier import NotifierConfig, WindowsNotifier

RECONNECT_BAD_FRAME_THRESHOLD = 5
RECONNECT_INTERVAL_SEC = 1.0
WINDOW_NAME = "Call Flash Detector"


def make_panel(image: np.ndarray | None, width: int, height: int, label: str) -> np.ndarray:
    if image is None:
        panel = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        panel = image
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        if panel.shape[:2] != (height, width):
            panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_LINEAR)

    cv2.putText(
        panel,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return panel


def is_bad_frame(ok: bool, frame: np.ndarray | None) -> bool:
    return (not ok) or frame is None or frame.size == 0 or (not frame.any())


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    use_dshow = os.name == "nt" and hasattr(cv2, "CAP_DSHOW")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_camera_fps(cap: cv2.VideoCapture, fallback_fps: int) -> float:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1.0:
        return float(fallback_fps)
    return fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect red-channel temporal spikes and notify user")

    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--fps", type=int, default=30)

    p.add_argument("--process-scale", type=float, default=1.0)
    p.add_argument("--cooldown", type=float, default=15.0)

    p.add_argument("--history-size", type=int, default=20)
    p.add_argument("--pixel-delta-threshold", type=int, default=28)
    p.add_argument("--min-active-pixels", type=int, default=30)
    p.add_argument("--activation-threshold", type=float, default=6.0)
    p.add_argument("--deactivation-threshold", type=float, default=2.5)
    p.add_argument("--on-frames", type=int, default=2)
    p.add_argument("--off-frames", type=int, default=3)

    p.add_argument("--show-preview", action="store_true")
    p.add_argument("--debug-fps", action="store_true")
    return p


def build_config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        process_scale=args.process_scale,
        cooldown=args.cooldown,
        history_size=args.history_size,
        pixel_delta_threshold=args.pixel_delta_threshold,
        min_active_pixels=args.min_active_pixels,
        activation_threshold=args.activation_threshold,
        deactivation_threshold=args.deactivation_threshold,
        on_frames=args.on_frames,
        off_frames=args.off_frames,
    )


def build_preview(
    frame: np.ndarray,
    detector: RedFlashDetector,
    status_line: str,
    is_active: bool,
) -> np.ndarray:
    overlay = frame.copy()

    bar_h = 34
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        status_line,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    led_color = (0, 255, 0) if is_active else (0, 0, 255)
    cv2.circle(overlay, (overlay.shape[1] - 20, 17), 6, led_color, -1)

    _, mask_raw, mask_clean, candidates, final = detector.get_last_stages()

    main_h, main_w = frame.shape[:2]
    side_w = max(220, main_w // 3)
    side_h = max(100, main_h // 3)

    left = overlay
    right_top = make_panel(final, side_w, side_h, "DETECTION")
    right_mid = make_panel(mask_raw, side_w, side_h, "RED CHANNEL")
    right_bottom = make_panel(candidates if candidates is not None else mask_clean, side_w, side_h, "DELTA")
    right = np.vstack((right_top, right_mid, right_bottom))

    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

    return np.hstack((left, right))


def run() -> None:
    args = build_parser().parse_args()

    config = build_config(args)
    detector = RedFlashDetector(config)
    notifier = WindowsNotifier(NotifierConfig(title="전화 감지 알림"))

    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. --camera-index 값을 확인하세요.")
    camera_fps = read_camera_fps(cap, args.fps)

    bad_count = 0
    last_reconnect = 0.0
    loop_frames = 0
    fps_start = time.perf_counter()

    if args.show_preview:
        cv2.namedWindow(WINDOW_NAME)

    try:
        while True:
            ok, frame = cap.read()
            now = time.perf_counter()

            if is_bad_frame(ok, frame):
                bad_count += 1
                if (
                    bad_count >= RECONNECT_BAD_FRAME_THRESHOLD
                    and now - last_reconnect >= RECONNECT_INTERVAL_SEC
                ):
                    last_reconnect = now
                    cap.release()
                    cap = open_camera(args.camera_index, args.width, args.height, args.fps)
                    detector = RedFlashDetector(config)
                    camera_fps = read_camera_fps(cap, args.fps)

                if args.show_preview:
                    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(
                        canvas,
                        "Camera signal lost. Reconnecting...",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, canvas)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                continue

            bad_count = 0
            metrics = detector.process(frame, now)

            if metrics.triggered:
                notifier.notify("전화가 감지되었습니다")

            if args.show_preview:
                state = "ACTIVE" if metrics.is_active else "IDLE"
                status = (
                    f"{state} | red={metrics.red_mean:.1f} base={metrics.baseline_mean:.1f} "
                    f"d_mean={metrics.delta_mean:.2f} d_px={metrics.delta_pixels} "
                    f"d_ratio={metrics.delta_ratio:.4f}"
                )
                preview = build_preview(frame, detector, status, metrics.is_active)
                cv2.imshow(WINDOW_NAME, preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.debug_fps:
                loop_frames += 1
                elapsed = now - fps_start
                if elapsed >= 1.0:
                    loop_fps = loop_frames / elapsed
                    print(f"FPS(raw={camera_fps:.1f}, loop={loop_fps:.1f})")
                    loop_frames = 0
                    fps_start = now

    finally:
        cap.release()
        if args.show_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
