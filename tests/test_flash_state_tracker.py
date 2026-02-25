from call_flash_detector.detector import DetectorConfig, FlashStateTracker


def test_rising_edge_triggers_once() -> None:
    cfg = DetectorConfig(cooldown=10.0)
    tracker = FlashStateTracker(cfg)

    active0, trig0 = tracker.update(False, 0.0)
    active1, trig1 = tracker.update(True, 0.2)
    active2, trig2 = tracker.update(True, 0.3)

    assert active0 is False and trig0 is False
    assert active1 is True and trig1 is True
    assert active2 is True and trig2 is False


def test_no_trigger_without_rising_edge() -> None:
    cfg = DetectorConfig(cooldown=10.0)
    tracker = FlashStateTracker(cfg)

    tracker.update(False, 0.0)
    tracker.update(False, 0.1)
    active, trig = tracker.update(False, 0.2)

    assert active is False
    assert trig is False


def test_cooldown_blocks_retrigger() -> None:
    cfg = DetectorConfig(cooldown=5.0)
    tracker = FlashStateTracker(cfg)

    tracker.update(False, 0.0)
    _, first = tracker.update(True, 0.1)

    tracker.update(False, 0.2)
    _, second = tracker.update(True, 1.0)

    tracker.update(False, 1.1)
    _, third = tracker.update(True, 6.2)

    assert first is True
    assert second is False
    assert third is True


def test_negative_cooldown_is_treated_as_zero() -> None:
    cfg = DetectorConfig(cooldown=-3.0)
    tracker = FlashStateTracker(cfg)

    tracker.update(False, 0.0)
    _, first = tracker.update(True, 0.1)
    tracker.update(False, 0.2)
    _, second = tracker.update(True, 0.3)

    assert first is True
    assert second is True
