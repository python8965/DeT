from __future__ import annotations

from dataclasses import dataclass
import sys


@dataclass(slots=True)
class NotifierConfig:
    title: str = "Incoming Call"


class WindowsNotifier:
    def __init__(self, config: NotifierConfig | None = None) -> None:
        self._cfg = config or NotifierConfig()

        try:
            from plyer import notification

            self._notification = notification
        except Exception as exc:
            print(
                f"[WARN] Failed to load plyer notification: {exc}",
                file=sys.stderr,
            )
            self._notification = None

    def notify(self, message: str) -> None:
        if self._notification is not None:
            try:
                self._notification.notify(
                    title=self._cfg.title,
                    message=message,
                    app_name="Call Flash Detector",
                    timeout=4,
                )
                return
            except Exception as exc:
                print(f"[ERROR] Failed to send notification: {exc}", file=sys.stderr)

        print(f"[NOTIFY] {self._cfg.title}: {message}")
