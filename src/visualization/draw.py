import cv2

# fixed colors per vehicle class so the same class always renders the same color
_CLASS_COLORS = {
    "car": (60, 200, 60),
    "truck": (60, 130, 240),
    "bus": (60, 60, 230),
    "motorcycle": (230, 160, 60),
    "bicycle": (200, 60, 200),
}
_DEFAULT_COLOR = (200, 200, 200)


def draw_tracked_object(frame, track_obj) -> None:
    x1, y1, x2, y2 = (int(v) for v in track_obj.xyxy)
    color = _CLASS_COLORS.get(track_obj.class_name, _DEFAULT_COLOR)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_obj.track_id} {track_obj.class_name} {track_obj.confidence:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_counting_line(frame, p1: tuple[float, float], p2: tuple[float, float]) -> None:
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))
    cv2.line(frame, pt1, pt2, (0, 220, 255), 3)


def draw_hud(frame, lines: list[str]) -> None:
    """Draws a semi-transparent stats panel in the top-left corner."""
    pad = 10
    line_h = 24
    width = 320
    height = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        y = pad + line_h * i + 18
        cv2.putText(frame, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
