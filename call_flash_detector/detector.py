from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

FIXED_HISTORY_SIZE = 10


@dataclass(slots=True)
class DetectorConfig:
    process_scale: float = 1.0
    cooldown: float = 1.0

    detect_size: int = 128
    color_delta_threshold: int = 18
    value_delta_threshold: int = 8
    min_active_pixels: int = 4
    activation_ratio: float = 0.00025
    deactivation_ratio: float = 0.00012
    scene_change_ignore_ratio: float = 0.5
    min_bright_value: int = 140
    max_active_ratio: float = 0.006

    # Shape filtering to reject camera-motion artifacts.
    min_shape_pixels: int = 2
    max_shape_ratio: float = 0.03
    max_shape_aspect_error: float = 0.75
    min_circle_circularity: float = 0.42
    min_rect_fill_ratio: float = 0.58
    min_shape_solidity: float = 0.72
    max_valid_shapes: int = 8
    small_shape_lenient_pixels: int = 12
    small_shape_min_fill_ratio: float = 0.34


@dataclass(slots=True)
class FrameMetrics:
    detect_size: int
    frame_mean: float
    baseline_mean: float
    delta_mean: float
    delta_pixels: int
    delta_ratio: float
    frame_area: int
    total_shape_count: int
    valid_shape_count: int
    rejected_shape_count: int
    scene_change_ratio: float
    scene_change_ignored: bool
    is_active: bool
    triggered: bool


class FlashStateTracker:
    """Tracks OFF->ON transition and enforces cooldown between notifications."""

    def __init__(self, config: DetectorConfig) -> None:
        self._cooldown = max(0.0, float(config.cooldown))
        self._last_state = False
        self._last_trigger_time = -10_000.0

    def update(self, is_active: bool, now: float) -> tuple[bool, bool]:
        triggered = False
        if is_active and not self._last_state:
            if now - self._last_trigger_time >= self._cooldown:
                triggered = True
                self._last_trigger_time = now

        self._last_state = is_active
        return is_active, triggered


class RedFlashDetector:
    """Detects LED on/off by full-frame color-change mask at low resolution."""

    def __init__(self, config: DetectorConfig) -> None:
        try:
            import cv2 as _cv2
            import numpy as _np
        except ModuleNotFoundError as exc:
            raise RuntimeError("opencv-python and numpy are required to use RedFlashDetector") from exc

        self._cv2 = _cv2
        self._np = _np
        self._cfg = self._normalize_config(config)
        self._tracker = FlashStateTracker(self._cfg)

        self._frame_sum: Any | None = None
        self._frame_history: deque[Any] = deque()
        self._last_shape: tuple[int, int, int] | None = None
        self._active_state = False

        self._last_overlay: Any | None = None
        self._last_red: Any | None = None
        self._last_delta: Any | None = None
        self._last_mask: Any | None = None
        self._morph_kernel = _np.ones((3, 3), dtype=_np.uint8)
        self._reject_colors: dict[str, tuple[int, int, int]] = {
            "too_small": (255, 120, 60),
            "too_large": (0, 0, 255),
            "aspect": (180, 70, 255),
            "solidity": (255, 60, 180),
            "shape": (120, 120, 120),
            "small_shape": (255, 255, 0),
            "too_many_shapes": (0, 165, 255),
        }

    @staticmethod
    def _normalize_config(config: DetectorConfig) -> DetectorConfig:
        process_scale = float(config.process_scale)
        if process_scale <= 0:
            process_scale = 1.0

        detect_size = max(32, min(512, int(config.detect_size)))

        activation_ratio = max(0.0, float(config.activation_ratio))
        deactivation_ratio = max(0.0, float(config.deactivation_ratio))
        deactivation_ratio = min(deactivation_ratio, activation_ratio)

        return replace(
            config,
            process_scale=process_scale,
            cooldown=max(0.0, float(config.cooldown)),
            detect_size=detect_size,
            color_delta_threshold=max(1, min(255, int(config.color_delta_threshold))),
            value_delta_threshold=max(0, min(255, int(config.value_delta_threshold))),
            min_active_pixels=max(1, int(config.min_active_pixels)),
            activation_ratio=activation_ratio,
            deactivation_ratio=deactivation_ratio,
            scene_change_ignore_ratio=max(0.05, min(1.0, float(config.scene_change_ignore_ratio))),
            min_bright_value=max(0, min(255, int(config.min_bright_value))),
            max_active_ratio=max(0.0001, min(1.0, float(config.max_active_ratio))),
            min_shape_pixels=max(1, int(config.min_shape_pixels)),
            max_shape_ratio=max(0.0001, min(1.0, float(config.max_shape_ratio))),
            max_shape_aspect_error=max(0.0, min(2.0, float(config.max_shape_aspect_error))),
            min_circle_circularity=max(0.0, min(1.0, float(config.min_circle_circularity))),
            min_rect_fill_ratio=max(0.0, min(1.0, float(config.min_rect_fill_ratio))),
            min_shape_solidity=max(0.0, min(1.0, float(config.min_shape_solidity))),
            max_valid_shapes=max(1, int(config.max_valid_shapes)),
            small_shape_lenient_pixels=max(1, int(config.small_shape_lenient_pixels)),
            small_shape_min_fill_ratio=max(0.0, min(1.0, float(config.small_shape_min_fill_ratio))),
        )

    def _update_history(self, frame_bgr: Any) -> None:
        if self._last_shape != tuple(frame_bgr.shape):
            self._frame_history.clear()
            self._frame_sum = None
            self._last_shape = tuple(frame_bgr.shape)

        frame_copy = frame_bgr.copy()
        if len(self._frame_history) >= FIXED_HISTORY_SIZE:
            oldest = self._frame_history.popleft()
            if self._frame_sum is not None:
                self._frame_sum = self._frame_sum - oldest

        self._frame_history.append(frame_copy)
        if self._frame_sum is None:
            self._frame_sum = frame_copy
        else:
            self._frame_sum = self._frame_sum + frame_copy

    def get_last_stages(self) -> tuple[Any | None, Any | None, Any | None, Any | None]:
        return (
            self._last_overlay,
            self._last_red,
            self._last_delta,
            self._last_mask,
        )

    def get_shape_filter_size(self) -> tuple[int, int]:
        return self._cfg.min_shape_pixels, self._cfg.small_shape_lenient_pixels

    def adjust_shape_filter_size(self, step: int) -> tuple[int, int]:
        if step == 0:
            return self.get_shape_filter_size()

        new_min = max(1, min(128, self._cfg.min_shape_pixels + int(step)))
        new_small = max(1, min(256, self._cfg.small_shape_lenient_pixels + (2 * int(step))))
        self._cfg = replace(
            self._cfg,
            min_shape_pixels=new_min,
            small_shape_lenient_pixels=new_small,
        )
        return self.get_shape_filter_size()

    def process(self, frame: Any, now: float, build_visuals: bool = False) -> FrameMetrics:
        cv2 = self._cv2
        np = self._np

        working = frame
        if self._cfg.process_scale != 1.0:
            working = cv2.resize(
                frame,
                dsize=None,
                fx=self._cfg.process_scale,
                fy=self._cfg.process_scale,
                interpolation=cv2.INTER_AREA,
            )

        detect_bgr = cv2.resize(
            working,
            (self._cfg.detect_size, self._cfg.detect_size),
            interpolation=cv2.INTER_AREA,
        )
        detect_bgr = cv2.GaussianBlur(detect_bgr, (3, 3), 0)
        detect = detect_bgr.astype(np.float32)

        if not self._frame_history:
            baseline = detect.copy()
        else:
            baseline = self._frame_sum / float(len(self._frame_history))

        delta_bgr = detect - baseline
        color_delta = np.max(delta_bgr, axis=2)
        color_delta_pos = np.maximum(color_delta, 0.0)

        curr_value = np.max(detect, axis=2)
        base_value = np.max(baseline, axis=2)
        value_delta = curr_value - base_value
        value_delta_pos = np.maximum(value_delta, 0.0)
        value_delta_abs = np.abs(value_delta)

        raw_mask = (
            (color_delta_pos >= float(self._cfg.color_delta_threshold))
            & (value_delta_pos >= float(self._cfg.value_delta_threshold))
            & (curr_value >= float(self._cfg.min_bright_value))
        ).astype(np.uint8)

        clean_mask = cv2.morphologyEx(raw_mask * 255, cv2.MORPH_OPEN, self._morph_kernel, iterations=1)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=1)
        contour_mask = (clean_mask > 0).astype(np.uint8)

        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_shape_count = len(contours)

        frame_area = max(1, contour_mask.size)
        scene_change_mask = value_delta_abs >= float(self._cfg.value_delta_threshold)
        scene_change_pixels = int(np.count_nonzero(scene_change_mask))
        scene_change_ratio = scene_change_pixels / float(frame_area)
        scene_change_ignored = scene_change_ratio >= self._cfg.scene_change_ignore_ratio
        max_shape_pixels = self._cfg.max_shape_ratio * float(frame_area)

        valid_mask = np.zeros_like(contour_mask, dtype=np.uint8)
        valid_shape_count = 0
        valid_boxes: list[tuple[int, int, int, int]] = []
        rejected_shapes: list[tuple[int, int, int, int, str]] = []

        for contour in contours:
            area_f = float(cv2.contourArea(contour))
            if area_f < float(self._cfg.min_shape_pixels) or area_f > max_shape_pixels:
                x, y, w, h = cv2.boundingRect(contour)
                reason = "too_small" if area_f < float(self._cfg.min_shape_pixels) else "too_large"
                rejected_shapes.append((x, y, w, h, reason))
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect_error = abs((w / float(h)) - 1.0)
            if aspect_error > self._cfg.max_shape_aspect_error:
                rejected_shapes.append((x, y, w, h, "aspect"))
                continue

            rect_fill = area_f / float(w * h)
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = float((4.0 * np.pi * area_f) / (perimeter * perimeter))
            circularity = max(0.0, min(1.0, circularity))

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            if hull_area <= 0.0:
                continue
            solidity = area_f / hull_area
            if solidity < self._cfg.min_shape_solidity:
                rejected_shapes.append((x, y, w, h, "solidity"))
                continue

            small_shape = area_f <= float(self._cfg.small_shape_lenient_pixels)
            if small_shape:
                if rect_fill < self._cfg.small_shape_min_fill_ratio:
                    rejected_shapes.append((x, y, w, h, "small_shape"))
                    continue
            else:
                approx = cv2.approxPolyDP(contour, 0.06 * perimeter, True)
                vertex_count = len(approx)
                circle_like = circularity >= self._cfg.min_circle_circularity and vertex_count >= 6
                rect_like = rect_fill >= self._cfg.min_rect_fill_ratio and 4 <= vertex_count <= 6
                if not (circle_like or rect_like):
                    rejected_shapes.append((x, y, w, h, "shape"))
                    continue

            valid_shape_count += 1
            valid_boxes.append((x, y, w, h))
            cv2.drawContours(valid_mask, [contour], -1, 255, thickness=-1)

        if valid_shape_count > self._cfg.max_valid_shapes:
            valid_shape_count = 0
            valid_boxes.clear()
            valid_mask.fill(0)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                rejected_shapes.append((x, y, w, h, "too_many_shapes"))

        active_mask = valid_mask > 0
        delta_pixels = int(np.count_nonzero(active_mask))
        delta_ratio = delta_pixels / float(frame_area)

        frame_mean = float(curr_value.mean())
        baseline_mean = float(base_value.mean())
        delta_mean = frame_mean - baseline_mean

        deactivate_pixels = max(1, self._cfg.min_active_pixels // 2)
        if scene_change_ignored or delta_ratio > self._cfg.max_active_ratio:
            self._active_state = False
        elif self._active_state:
            if delta_pixels < deactivate_pixels or delta_ratio <= self._cfg.deactivation_ratio:
                self._active_state = False
        elif (
            valid_shape_count > 0
            and delta_pixels >= self._cfg.min_active_pixels
            and delta_ratio >= self._cfg.activation_ratio
            and delta_ratio <= self._cfg.max_active_ratio
        ):
            self._active_state = True

        history_ready = len(self._frame_history) >= FIXED_HISTORY_SIZE
        is_active, triggered = self._tracker.update(
            history_ready and self._active_state and (not scene_change_ignored),
            now,
        )

        self._update_history(detect)

        if build_visuals:
            value_vis = cv2.normalize(curr_value, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_delta_vis = cv2.normalize(color_delta_pos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            overlay = working.copy()
            sx = working.shape[1] / float(self._cfg.detect_size)
            sy = working.shape[0] / float(self._cfg.detect_size)
            for x, y, w, h in valid_boxes:
                x1 = int(x * sx)
                y1 = int(y * sy)
                x2 = int((x + w) * sx) - 1
                y2 = int((y + h) * sy) - 1
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            for x, y, w, h, reason in rejected_shapes:
                x1 = int(x * sx)
                y1 = int(y * sy)
                x2 = int((x + w) * sx) - 1
                y2 = int((y + h) * sy) - 1
                color = self._reject_colors.get(reason, (100, 100, 100))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

            self._last_overlay = overlay
            self._last_red = cv2.applyColorMap(value_vis, cv2.COLORMAP_HOT)
            self._last_delta = cv2.applyColorMap(color_delta_vis, cv2.COLORMAP_JET)
            self._last_mask = cv2.cvtColor(valid_mask, cv2.COLOR_GRAY2BGR)

        return FrameMetrics(
            detect_size=self._cfg.detect_size,
            frame_mean=frame_mean,
            baseline_mean=baseline_mean,
            delta_mean=delta_mean,
            delta_pixels=delta_pixels,
            delta_ratio=delta_ratio,
            frame_area=frame_area,
            total_shape_count=total_shape_count,
            valid_shape_count=valid_shape_count,
            rejected_shape_count=len(rejected_shapes),
            scene_change_ratio=scene_change_ratio,
            scene_change_ignored=scene_change_ignored,
            is_active=is_active,
            triggered=triggered,
        )
