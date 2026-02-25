from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(slots=True)
class DetectorConfig:
    process_scale: float = 1.0
    cooldown: float = 15.0

    # Temporal red-channel model
    history_size: int = 20
    pixel_delta_threshold: int = 28
    min_active_pixels: int = 30
    activation_threshold: float = 6.0
    deactivation_threshold: float = 2.5

    # State smoothing
    on_frames: int = 2
    off_frames: int = 3


@dataclass(slots=True)
class FrameMetrics:
    red_mean: float
    baseline_mean: float
    delta_mean: float
    delta_pixels: int
    delta_ratio: float
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
    """Detects red LED-like events from temporal red-channel change only."""

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

        self._red_baseline: Any | None = None
        self._stable_active = False
        self._on_streak = 0
        self._off_streak = 0

        self._last_blurred: Any | None = None
        self._last_mask_raw: Any | None = None
        self._last_mask_clean: Any | None = None
        self._last_candidates: Any | None = None
        self._last_processed: Any | None = None

    @staticmethod
    def _normalize_config(config: DetectorConfig) -> DetectorConfig:
        process_scale = float(config.process_scale)
        if process_scale <= 0:
            process_scale = 1.0

        history_size = max(2, int(config.history_size))
        min_active_pixels = max(1, int(config.min_active_pixels))

        return replace(
            config,
            process_scale=process_scale,
            cooldown=max(0.0, float(config.cooldown)),
            history_size=history_size,
            pixel_delta_threshold=max(1, min(255, int(config.pixel_delta_threshold))),
            min_active_pixels=min_active_pixels,
            activation_threshold=max(0.0, float(config.activation_threshold)),
            deactivation_threshold=max(0.0, float(config.deactivation_threshold)),
            on_frames=max(1, int(config.on_frames)),
            off_frames=max(1, int(config.off_frames)),
        )

    def get_last_stages(
        self,
    ) -> tuple[Any | None, Any | None, Any | None, Any | None, Any | None]:
        return (
            self._last_blurred,
            self._last_mask_raw,
            self._last_mask_clean,
            self._last_candidates,
            self._last_processed,
        )

    def _baseline_alpha(self) -> float:
        # Rolling-average approximation of N-frame mean.
        return 1.0 / float(self._cfg.history_size)

    def process(self, frame: Any, now: float) -> FrameMetrics:
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

        blurred = cv2.GaussianBlur(working, (5, 5), 0)
        red = blurred[:, :, 2].astype(np.float32)

        if self._red_baseline is None:
            self._red_baseline = red.copy()

        red_delta = red - self._red_baseline
        delta_mask = red_delta >= float(self._cfg.pixel_delta_threshold)
        delta_pixels = int(np.count_nonzero(delta_mask))
        total_pixels = max(1, delta_mask.size)
        delta_ratio = delta_pixels / float(total_pixels)

        red_mean = float(red.mean())
        baseline_mean = float(self._red_baseline.mean())
        delta_mean = red_mean - baseline_mean

        candidate_on = (
            delta_mean >= self._cfg.activation_threshold
            and delta_pixels >= self._cfg.min_active_pixels
        )

        if candidate_on:
            self._on_streak += 1
            self._off_streak = 0
        elif delta_mean <= self._cfg.deactivation_threshold:
            self._off_streak += 1
            self._on_streak = 0

        if not self._stable_active and self._on_streak >= self._cfg.on_frames:
            self._stable_active = True
        elif self._stable_active and self._off_streak >= self._cfg.off_frames:
            self._stable_active = False

        is_active, triggered = self._tracker.update(self._stable_active, now)

        # Always update baseline so static red scene converges to IDLE.
        alpha = self._baseline_alpha()
        self._red_baseline = ((1.0 - alpha) * self._red_baseline) + (alpha * red)

        red_vis = cv2.normalize(red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        delta_vis = cv2.normalize(np.maximum(red_delta, 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_vis = delta_mask.astype(np.uint8) * 255

        processed = working.copy()
        color = (0, 255, 0) if is_active else (0, 0, 255)
        cv2.putText(
            processed,
            f"delta_mean={delta_mean:.2f} delta_pixels={delta_pixels}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

        self._last_blurred = blurred
        self._last_mask_raw = cv2.applyColorMap(red_vis, cv2.COLORMAP_HOT)
        self._last_mask_clean = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
        self._last_candidates = cv2.applyColorMap(delta_vis, cv2.COLORMAP_JET)
        self._last_processed = processed

        return FrameMetrics(
            red_mean=red_mean,
            baseline_mean=baseline_mean,
            delta_mean=delta_mean,
            delta_pixels=delta_pixels,
            delta_ratio=delta_ratio,
            is_active=is_active,
            triggered=triggered,
        )
