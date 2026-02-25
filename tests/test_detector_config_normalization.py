from call_flash_detector.detector import DetectorConfig, RedFlashDetector


def test_normalize_config_clamps_runtime_fields() -> None:
    cfg = DetectorConfig(
        process_scale=-1.0,
        cooldown=-3.0,
        history_size=0,
        pixel_delta_threshold=999,
        min_active_pixels=-10,
        activation_threshold=-1.0,
        deactivation_threshold=-1.0,
        on_frames=0,
        off_frames=-2,
    )

    out = RedFlashDetector._normalize_config(cfg)

    assert out.process_scale == 1.0
    assert out.cooldown == 0.0
    assert out.history_size == 2
    assert out.pixel_delta_threshold == 255
    assert out.min_active_pixels == 1
    assert out.activation_threshold == 0.0
    assert out.deactivation_threshold == 0.0
    assert out.on_frames == 1
    assert out.off_frames == 1
