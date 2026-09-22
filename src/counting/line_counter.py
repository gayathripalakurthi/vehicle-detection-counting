import math
from dataclasses import dataclass


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str  # "IN" or "OUT"
    frame_idx: int


@dataclass
class _TrackState:
    confirmed_side: int | None = None  # -1 or +1, only set once past the margin
    counted: bool = False


class LineCounter:
    """Counts tracked vehicles crossing a configured line, once per track ID.

    Side is determined via the sign of the 2D cross product of the line
    vector and the vector from the line's start point to the vehicle's
    ground-contact point (bottom-center of its box) - the standard anchor
    for vehicle counting since it tracks where the vehicle touches the road,
    not its centroid, which drifts with vehicle height/perspective.

    A vehicle sitting near the line for a few frames produces a noisy,
    jittering box (especially at borderline confidence), which can flip
    the raw sign back and forth without a real crossing happening - e.g.
    two cars both driving the same direction got opposite IN/OUT labels
    in testing because one lingered right at the boundary. To guard
    against this, a side is only "confirmed" once the vehicle's anchor
    point is more than `margin_px` past the line; readings inside that
    dead zone are ignored rather than treated as a side change.
    """

    def __init__(self, config: dict, frame_width: int, frame_height: int):
        line_cfg = config["counting"]["line"]
        p1 = line_cfg["point1"]
        p2 = line_cfg["point2"]
        self.p1 = (p1[0] * frame_width, p1[1] * frame_height)
        self.p2 = (p2[0] * frame_width, p2[1] * frame_height)
        self.in_label, self.out_label = config["counting"]["direction_labels"]
        self.margin_px = line_cfg.get("margin_px", 15)

        (x1, y1), (x2, y2) = self.p1, self.p2
        self._line_length = math.hypot(x2 - x1, y2 - y1)

        self._states: dict[int, _TrackState] = {}
        self.counts: dict[str, dict[str, int]] = {}  # class_name -> {IN: n, OUT: n}

    def _signed_distance(self, point: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return cross / self._line_length

    @staticmethod
    def _anchor_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2, y2)

    def update(self, tracked_objects, frame_idx: int) -> list[CrossingEvent]:
        events = []
        for obj in tracked_objects:
            distance = self._signed_distance(self._anchor_point(obj.xyxy))
            state = self._states.setdefault(obj.track_id, _TrackState())

            if abs(distance) <= self.margin_px:
                continue  # ambiguous - too close to the line to trust this frame

            side = 1 if distance > 0 else -1

            if state.confirmed_side is not None and not state.counted and side != state.confirmed_side:
                direction = self.in_label if side < 0 else self.out_label
                state.counted = True
                events.append(CrossingEvent(obj.track_id, obj.class_name, direction, frame_idx))
                class_counts = self.counts.setdefault(obj.class_name, {self.in_label: 0, self.out_label: 0})
                class_counts[direction] += 1

            state.confirmed_side = side

        return events

    def distance_to_line(self, xyxy: tuple[float, float, float, float]) -> float:
        return self._signed_distance(self._anchor_point(xyxy))

    def is_counted(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state is not None and state.counted

    def total(self) -> int:
        return sum(sum(d.values()) for d in self.counts.values())

    def totals_by_direction(self) -> dict[str, int]:
        totals = {self.in_label: 0, self.out_label: 0}
        for class_counts in self.counts.values():
            for direction, n in class_counts.items():
                totals[direction] += n
        return totals
