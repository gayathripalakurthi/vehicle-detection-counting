from dataclasses import dataclass, field


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str  # "IN" or "OUT"
    frame_idx: int


@dataclass
class _TrackState:
    last_side: float | None = None
    counted: bool = False


class LineCounter:
    """Counts tracked vehicles crossing a configured line, once per track ID.

    Side is determined via the sign of the 2D cross product of the line
    vector and the vector from the line's start point to the vehicle's
    ground-contact point (bottom-center of its box) - the standard anchor
    for vehicle counting since it tracks where the vehicle touches the road,
    not its centroid, which drifts with vehicle height/perspective.
    """

    def __init__(self, config: dict, frame_width: int, frame_height: int):
        line_cfg = config["counting"]["line"]
        p1 = line_cfg["point1"]
        p2 = line_cfg["point2"]
        self.p1 = (p1[0] * frame_width, p1[1] * frame_height)
        self.p2 = (p2[0] * frame_width, p2[1] * frame_height)
        self.in_label, self.out_label = config["counting"]["direction_labels"]

        self._states: dict[int, _TrackState] = {}
        self.counts: dict[str, dict[str, int]] = {}  # class_name -> {IN: n, OUT: n}

    def _side(self, point: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.p1, self.p2
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    @staticmethod
    def _anchor_point(xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = xyxy
        return ((x1 + x2) / 2, y2)

    def update(self, tracked_objects, frame_idx: int) -> list[CrossingEvent]:
        events = []
        for obj in tracked_objects:
            side = self._side(self._anchor_point(obj.xyxy))
            state = self._states.setdefault(obj.track_id, _TrackState())

            if state.last_side is not None and not state.counted and side != 0:
                crossed_to_negative = state.last_side > 0 and side < 0
                crossed_to_positive = state.last_side < 0 and side > 0
                if crossed_to_negative or crossed_to_positive:
                    direction = self.in_label if crossed_to_negative else self.out_label
                    state.counted = True
                    events.append(CrossingEvent(obj.track_id, obj.class_name, direction, frame_idx))
                    class_counts = self.counts.setdefault(obj.class_name, {self.in_label: 0, self.out_label: 0})
                    class_counts[direction] += 1

            if side != 0:
                state.last_side = side

        return events

    def total(self) -> int:
        return sum(sum(d.values()) for d in self.counts.values())

    def totals_by_direction(self) -> dict[str, int]:
        totals = {self.in_label: 0, self.out_label: 0}
        for class_counts in self.counts.values():
            for direction, n in class_counts.items():
                totals[direction] += n
        return totals
