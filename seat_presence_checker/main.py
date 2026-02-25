from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

WINDOW_NAME = "Seat Presence Checker"
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0
ZOOM_STEP = 0.1
PAN_STEP = 40


@dataclass(slots=True)
class SeatBox:
    seat_id: int
    name: str
    x: int
    y: int


@dataclass(slots=True)
class LayoutData:
    canvas_width: int
    canvas_height: int
    seat_width: int
    seat_height: int
    seats: list[SeatBox]

    @staticmethod
    def load(path: Path) -> "LayoutData":
        raw = json.loads(path.read_text(encoding="utf-8"))
        seats: list[SeatBox] = []
        for row in raw.get("seats", []):
            seats.append(
                SeatBox(
                    seat_id=int(row["seat_id"]),
                    name=str(row.get("name", f"SEAT-{int(row['seat_id'])}")),
                    x=int(row["x"]),
                    y=int(row["y"]),
                )
            )
        return LayoutData(
            canvas_width=int(raw.get("canvas_width", 1280)),
            canvas_height=int(raw.get("canvas_height", 720)),
            seat_width=max(12, int(raw.get("seat_width", 140))),
            seat_height=max(12, int(raw.get("seat_height", 90))),
            seats=seats,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "seat_width": self.seat_width,
            "seat_height": self.seat_height,
            "seats": [asdict(s) for s in self.seats],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(slots=True)
class SeatState:
    present: bool
    away_since: float | None


def make_canvas(w: int, h: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    for yy in range(0, h, 40):
        cv2.line(canvas, (0, yy), (w, yy), (34, 34, 34), 1)
    for xx in range(0, w, 40):
        cv2.line(canvas, (xx, 0), (xx, h), (34, 34, 34), 1)
    return canvas


def clamp_seat_xy(layout: LayoutData, x: int, y: int) -> tuple[int, int]:
    x = min(max(0, x), max(0, layout.canvas_width - layout.seat_width))
    y = min(max(0, y), max(0, layout.canvas_height - layout.seat_height))
    return x, y


def seat_rect(layout: LayoutData, seat: SeatBox) -> tuple[int, int, int, int]:
    return seat.x, seat.y, layout.seat_width, layout.seat_height


def hit_test(layout: LayoutData, x: int, y: int) -> SeatBox | None:
    for seat in reversed(layout.seats):
        sx, sy, sw, sh = seat_rect(layout, seat)
        if sx <= x <= sx + sw and sy <= y <= sy + sh:
            return seat
    return None


def next_id(layout: LayoutData) -> int:
    if not layout.seats:
        return 1
    return max(s.seat_id for s in layout.seats) + 1


def load_state(path: Path, layout: LayoutData) -> dict[int, SeatState]:
    states = {s.seat_id: SeatState(present=False, away_since=None) for s in layout.seats}
    if not path.exists():
        return states

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("states", {})
    now = time.time()
    for seat in layout.seats:
        row = rows.get(str(seat.seat_id))
        if not isinstance(row, dict):
            continue
        present = bool(row.get("present", False))
        away_since = row.get("away_since")
        if away_since is None:
            away = None
        else:
            away = float(away_since)
            if away > now + 10_000:
                away = now
        states[seat.seat_id] = SeatState(present=present, away_since=away)
    return states


def save_state(path: Path, states: dict[int, SeatState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": time.time(),
        "states": {
            str(k): {
                "present": v.present,
                "away_since": v.away_since,
            }
            for k, v in states.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_states(states: dict[int, SeatState], layout: LayoutData) -> dict[int, SeatState]:
    out: dict[int, SeatState] = {}
    for seat in layout.seats:
        st = states.get(seat.seat_id)
        if st is None:
            st = SeatState(present=False, away_since=None)
        out[seat.seat_id] = st
    return out


def fit_label_text(
    text: str,
    max_width: int,
    *,
    preferred_scale: float = 0.54,
    min_scale: float = 0.32,
    thickness: int = 1,
) -> tuple[str, float]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap_width = max(12, int(max_width))
    scale = preferred_scale
    while scale >= min_scale:
        (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if text_w <= cap_width:
            return text, scale
        scale -= 0.04

    ellipsis = "..."
    trimmed = text
    while trimmed:
        candidate = f"{trimmed}{ellipsis}"
        (text_w, _), _ = cv2.getTextSize(candidate, font, min_scale, thickness)
        if text_w <= cap_width:
            return candidate, min_scale
        trimmed = trimmed[:-1]
    return ellipsis, min_scale


def clamp_pan(layout: LayoutData, zoom: float, pan_x: int, pan_y: int, view_w: int, view_h: int) -> tuple[int, int]:
    scaled_w = max(1, int(round(layout.canvas_width * zoom)))
    scaled_h = max(1, int(round(layout.canvas_height * zoom)))
    max_x = max(0, scaled_w - view_w)
    max_y = max(0, scaled_h - view_h)
    return min(max(0, pan_x), max_x), min(max(0, pan_y), max_y)


def render_viewport(frame: np.ndarray, layout: LayoutData, zoom: float, pan_x: int, pan_y: int, view_w: int, view_h: int) -> np.ndarray:
    scaled_w = max(1, int(round(layout.canvas_width * zoom)))
    scaled_h = max(1, int(round(layout.canvas_height * zoom)))
    scaled = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    if scaled_w >= view_w and scaled_h >= view_h:
        return scaled[pan_y : pan_y + view_h, pan_x : pan_x + view_w]

    out = np.zeros((view_h, view_w, 3), dtype=np.uint8)
    ox = max(0, (view_w - scaled_w) // 2)
    oy = max(0, (view_h - scaled_h) // 2)
    x2 = min(view_w, ox + scaled_w)
    y2 = min(view_h, oy + scaled_h)
    out[oy:y2, ox:x2] = scaled[: y2 - oy, : x2 - ox]
    return out


def view_to_canvas(
    x: int,
    y: int,
    layout: LayoutData,
    zoom: float,
    pan_x: int,
    pan_y: int,
    view_w: int,
    view_h: int,
) -> tuple[int, int] | None:
    scaled_w = max(1, int(round(layout.canvas_width * zoom)))
    scaled_h = max(1, int(round(layout.canvas_height * zoom)))

    if scaled_w >= view_w:
        sx = x + pan_x
    else:
        ox = (view_w - scaled_w) // 2
        sx = x - ox
    if scaled_h >= view_h:
        sy = y + pan_y
    else:
        oy = (view_h - scaled_h) // 2
        sy = y - oy

    if sx < 0 or sy < 0 or sx >= scaled_w or sy >= scaled_h:
        return None

    cx = int(sx / zoom)
    cy = int(sy / zoom)
    cx = min(max(0, cx), layout.canvas_width - 1)
    cy = min(max(0, cy), layout.canvas_height - 1)
    return cx, cy


def draw(layout: LayoutData, states: dict[int, SeatState], mode: str, zoom: float, message: str = "") -> np.ndarray:
    canvas = make_canvas(layout.canvas_width, layout.canvas_height)

    present_count = 0
    for seat in layout.seats:
        sx, sy, sw, sh = seat_rect(layout, seat)
        st = states.get(seat.seat_id, SeatState(False, None))

        if mode == "edit":
            color = (0, 255, 255)
            label = seat.name
        else:
            if st.present:
                present_count += 1
                color = (0, 220, 0)
                label = f"{seat.name}: IN"
            else:
                color = (0, 165, 255)
                label = f"{seat.name}: AWAY"

        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), color, 2)

        draw_label, label_scale = fit_label_text(label, sw - 10)
        (tw, th), base = cv2.getTextSize(draw_label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        tx = sx + max(4, (sw - tw) // 2)
        ty = sy + min(sh - 4, max(th + 4, (sh + th) // 2))

        bg_x1 = max(sx + 1, tx - 2)
        bg_y1 = max(sy + 1, ty - th - 2)
        bg_x2 = min(sx + sw - 1, tx + tw + 2)
        bg_y2 = min(sy + sh - 1, ty + base + 2)
        cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), (20, 20, 20), -1)
        cv2.putText(canvas, draw_label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), 1, cv2.LINE_AA)

    if mode == "edit":
        header = "MODE=EDIT | L=place seat | R=delete | M-drag/HJKL=pan | +/- zoom 0=reset | TAB/m=RUN | s=save | q/x/esc"
    else:
        away_count = len(layout.seats) - present_count
        header = (
            f"MODE=RUN | L=toggle IN/AWAY | M-drag/HJKL=pan | +/- zoom 0=reset | TAB/m=EDIT | p={present_count} away={away_count} | s=save | q/x/esc"
        )

    cv2.putText(canvas, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"seat_size={layout.seat_width}x{layout.seat_height} count={len(layout.seats)} zoom={zoom:.2f}x",
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    if message:
        cv2.putText(canvas, message, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 0), 2, cv2.LINE_AA)
    return canvas


def run_app(layout_path: Path, state_path: Path, width: int, height: int, seat_width: int, seat_height: int, start_mode: str) -> None:
    if layout_path.exists():
        layout = LayoutData.load(layout_path)
        if width > 0:
            layout.canvas_width = width
        if height > 0:
            layout.canvas_height = height
        if seat_width > 0:
            layout.seat_width = seat_width
        if seat_height > 0:
            layout.seat_height = seat_height
        layout.seat_width = max(12, layout.seat_width)
        layout.seat_height = max(12, layout.seat_height)
    else:
        layout = LayoutData(
            canvas_width=max(320, width),
            canvas_height=max(240, height),
            seat_width=max(12, seat_width),
            seat_height=max(12, seat_height),
            seats=[],
        )

    states = load_state(state_path, layout)
    states = sync_states(states, layout)

    class UiState:
        def __init__(self) -> None:
            self.mode = start_mode if start_mode in ("edit", "run") else "edit"
            self.left_click: tuple[int, int] | None = None
            self.right_click: tuple[int, int] | None = None
            self.message = ""
            self.msg_until = 0.0
            self.zoom = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.is_panning = False
            self.last_mouse: tuple[int, int] | None = None

    ui = UiState()
    view_w = layout.canvas_width
    view_h = layout.canvas_height

    def set_zoom(new_zoom: float) -> None:
        old_zoom = max(0.01, ui.zoom)
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, float(new_zoom)))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        center_canvas_x = (ui.pan_x + (view_w / 2.0)) / old_zoom
        center_canvas_y = (ui.pan_y + (view_h / 2.0)) / old_zoom
        ui.zoom = new_zoom
        ui.pan_x = int(center_canvas_x * ui.zoom - (view_w / 2.0))
        ui.pan_y = int(center_canvas_y * ui.zoom - (view_h / 2.0))
        ui.pan_x, ui.pan_y = clamp_pan(layout, ui.zoom, ui.pan_x, ui.pan_y, view_w, view_h)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_MBUTTONDOWN:
            ui.is_panning = True
            ui.last_mouse = (x, y)
            return
        if event == cv2.EVENT_MBUTTONUP:
            ui.is_panning = False
            ui.last_mouse = None
            return
        if event == cv2.EVENT_MOUSEMOVE and ui.is_panning and ui.last_mouse is not None:
            px, py = ui.last_mouse
            dx = x - px
            dy = y - py
            ui.pan_x -= dx
            ui.pan_y -= dy
            ui.pan_x, ui.pan_y = clamp_pan(layout, ui.zoom, ui.pan_x, ui.pan_y, view_w, view_h)
            ui.last_mouse = (x, y)
            return

        pt = view_to_canvas(x, y, layout, ui.zoom, ui.pan_x, ui.pan_y, view_w, view_h)
        if pt is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            ui.left_click = pt
        elif event == cv2.EVENT_RBUTTONDOWN:
            ui.right_click = pt

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, view_w, view_h)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            if time.time() > ui.msg_until:
                ui.message = ""

            if ui.left_click is not None:
                x, y = ui.left_click
                ui.left_click = None

                if ui.mode == "edit":
                    hit = hit_test(layout, x, y)
                    if hit is None:
                        nx, ny = clamp_seat_xy(layout, x - (layout.seat_width // 2), y - (layout.seat_height // 2))
                        sid = next_id(layout)
                        layout.seats.append(SeatBox(seat_id=sid, name=f"SEAT-{sid}", x=nx, y=ny))
                        states = sync_states(states, layout)
                else:
                    hit = hit_test(layout, x, y)
                    if hit is not None:
                        st = states[hit.seat_id]
                        st.present = not st.present
                        st.away_since = None if st.present else time.time()
                        save_state(state_path, states)

            if ui.right_click is not None:
                x, y = ui.right_click
                ui.right_click = None
                if ui.mode == "edit":
                    hit = hit_test(layout, x, y)
                    if hit is not None:
                        layout.seats = [s for s in layout.seats if s.seat_id != hit.seat_id]
                        states = sync_states(states, layout)

            ui.pan_x, ui.pan_y = clamp_pan(layout, ui.zoom, ui.pan_x, ui.pan_y, view_w, view_h)
            frame = draw(layout, states, ui.mode, ui.zoom, ui.message)
            show_frame = render_viewport(frame, layout, ui.zoom, ui.pan_x, ui.pan_y, view_w, view_h)
            cv2.imshow(WINDOW_NAME, show_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q"), ord("x"), ord("X")):
                break
            if key in (9, ord("m"), ord("M")):
                ui.mode = "run" if ui.mode == "edit" else "edit"
                ui.message = f"mode switched -> {ui.mode.upper()}"
                ui.msg_until = time.time() + 1.2
            elif key in (ord("s"), ord("S")):
                layout.save(layout_path)
                save_state(state_path, states)
                ui.message = f"saved: {layout_path.name}, {state_path.name}"
                ui.msg_until = time.time() + 1.5
            elif key in (ord("c"), ord("C")) and ui.mode == "edit":
                layout.seats.clear()
                states = sync_states(states, layout)
            elif key in (ord("+"), ord("="), ord("]")):
                set_zoom(round(ui.zoom + ZOOM_STEP, 2))
                ui.message = f"zoom={ui.zoom:.2f}x"
                ui.msg_until = time.time() + 1.0
            elif key in (ord("-"), ord("_"), ord("[")):
                set_zoom(round(ui.zoom - ZOOM_STEP, 2))
                ui.message = f"zoom={ui.zoom:.2f}x"
                ui.msg_until = time.time() + 1.0
            elif key == ord("0"):
                set_zoom(1.0)
                ui.pan_x = 0
                ui.pan_y = 0
                ui.message = "zoom reset to 1.00x"
                ui.msg_until = time.time() + 1.0
            elif key in (ord("h"), ord("H")):
                ui.pan_x -= PAN_STEP
            elif key in (ord("l"), ord("L")):
                ui.pan_x += PAN_STEP
            elif key in (ord("k"), ord("K")):
                ui.pan_y -= PAN_STEP
            elif key in (ord("j"), ord("J")):
                ui.pan_y += PAN_STEP

    finally:
        layout.save(layout_path)
        save_state(state_path, states)
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="클릭 기반 자리 상태 체크 (단일 앱, edit/run 전환)")
    p.add_argument("--layout", type=Path, default=Path("data/seat_presence_checker/layout.json"))
    p.add_argument("--state", type=Path, default=Path("data/seat_presence_checker/state.json"))
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--seat-width", type=int, default=140)
    p.add_argument("--seat-height", type=int, default=90)
    p.add_argument("--start-mode", choices=("edit", "run"), default="edit")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_app(
        layout_path=args.layout,
        state_path=args.state,
        width=args.width,
        height=args.height,
        seat_width=args.seat_width,
        seat_height=args.seat_height,
        start_mode=args.start_mode,
    )


if __name__ == "__main__":
    main()
