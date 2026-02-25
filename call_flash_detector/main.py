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
    return (not ok) or frame is None or frame.size == 0


def center_square_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


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
    p.add_argument("--cooldown", type=float, default=5.0)

    p.add_argument("--detect-size", type=int, default=128)
    p.add_argument("--color-delta-threshold", "--pixel-delta-threshold", dest="color_delta_threshold", type=int, default=18)
    p.add_argument("--value-delta-threshold", type=int, default=8)
    p.add_argument("--min-active-pixels", type=int, default=4)
    p.add_argument("--activation-ratio", type=float, default=0.00025)
    p.add_argument("--deactivation-ratio", type=float, default=0.00012)
    p.add_argument("--scene-change-ignore-ratio", type=float, default=0.5)
    p.add_argument("--min-bright-value", type=int, default=140)
    p.add_argument("--max-active-ratio", type=float, default=0.006)
    p.add_argument("--min-shape-pixels", type=int, default=2)
    p.add_argument("--max-shape-ratio", type=float, default=0.03)
    p.add_argument("--max-shape-aspect-error", type=float, default=0.75)
    p.add_argument("--min-circle-circularity", type=float, default=0.42)
    p.add_argument("--min-rect-fill-ratio", type=float, default=0.58)
    p.add_argument("--min-shape-solidity", type=float, default=0.72)
    p.add_argument("--max-valid-shapes", type=int, default=8)
    p.add_argument("--small-shape-lenient-pixels", type=int, default=12)
    p.add_argument("--small-shape-min-fill-ratio", type=float, default=0.34)

    p.add_argument("--show-preview", action="store_true")
    p.add_argument("--debug-fps", action="store_true")
    return p


def build_config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
        process_scale=args.process_scale,
        cooldown=args.cooldown,
        detect_size=args.detect_size,
        color_delta_threshold=args.color_delta_threshold,
        value_delta_threshold=args.value_delta_threshold,
        min_active_pixels=args.min_active_pixels,
        activation_ratio=args.activation_ratio,
        deactivation_ratio=args.deactivation_ratio,
        scene_change_ignore_ratio=args.scene_change_ignore_ratio,
        min_bright_value=args.min_bright_value,
        max_active_ratio=args.max_active_ratio,
        min_shape_pixels=args.min_shape_pixels,
        max_shape_ratio=args.max_shape_ratio,
        max_shape_aspect_error=args.max_shape_aspect_error,
        min_circle_circularity=args.min_circle_circularity,
        min_rect_fill_ratio=args.min_rect_fill_ratio,
        min_shape_solidity=args.min_shape_solidity,
        max_valid_shapes=args.max_valid_shapes,
        small_shape_lenient_pixels=args.small_shape_lenient_pixels,
        small_shape_min_fill_ratio=args.small_shape_min_fill_ratio,
    )


def build_preview(
    frame: np.ndarray,
    detector: RedFlashDetector,
    status_line: str,
    is_active: bool,
    scene_change_ignored: bool,
) -> np.ndarray:
    overlay, _, frame_delta, frame_mask = detector.get_last_stages()
    if overlay is None:
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

    if scene_change_ignored:
        led_color = (0, 165, 255)
        cv2.putText(
            overlay,
            "BIG CHANGE: NOTIFY BLOCKED",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        led_color = (0, 255, 0) if is_active else (0, 0, 255)
    cv2.circle(overlay, (overlay.shape[1] - 20, 17), 6, led_color, -1)

    main_h, main_w = overlay.shape[:2]
    side_w = max(220, main_w // 3)
    side_h = max(120, main_h // 2)

    left = overlay
    right_top = make_panel(frame_delta, side_w, side_h, "DELTA+")
    right_bottom = make_panel(frame_mask, side_w, side_h, "VALID MASK")
    right = np.vstack((right_top, right_bottom))

    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR)

    return np.hstack((left, right))


def should_close_preview_window() -> bool:
    try:
        visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        autosize = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_AUTOSIZE)
        return visible < 1 or autosize < 0
    except cv2.error:
        return True


def handle_preview_key(detector: RedFlashDetector | None) -> bool:
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
        return True

    if detector is not None:
        if key in (ord("="), ord("+"), ord("]")):
            detector.adjust_shape_filter_size(1)
        elif key in (ord("-"), ord("_"), ord("[")):
            detector.adjust_shape_filter_size(-1)
    return False


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
                    if should_close_preview_window() or handle_preview_key(None):
                        break
                else:
                    # Avoid tight spin when camera input is unavailable.
                    time.sleep(0.03)
                continue

            bad_count = 0
            frame_square = center_square_crop(frame)
            metrics = detector.process(frame_square, now, build_visuals=args.show_preview)

            if metrics.triggered:
                notifier.notify("전화가 감지되었습니다")

            if args.show_preview:
                if metrics.scene_change_ignored:
                    state = "IGNORED(BIG_CHANGE)"
                else:
                    state = "ACTIVE" if metrics.is_active else "IDLE"
                min_shape_px, lenient_px = detector.get_shape_filter_size()
                status = (
                    f"{state} | d_ratio={metrics.delta_ratio:.4f} "
                    f"scene={metrics.scene_change_ratio:.2f} "
                    f"shape={metrics.valid_shape_count}/{metrics.total_shape_count} "
                    f"rej={metrics.rejected_shape_count} trig={int(metrics.triggered)} "
                    f"fsize={min_shape_px}/{lenient_px}"
                )
                preview = build_preview(
                    frame_square,
                    detector,
                    status,
                    metrics.is_active,
                    metrics.scene_change_ignored,
                )
                cv2.imshow(WINDOW_NAME, preview)
                if should_close_preview_window() or handle_preview_key(detector):
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
