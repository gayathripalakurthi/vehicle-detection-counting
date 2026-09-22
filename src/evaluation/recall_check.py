from dataclasses import dataclass


@dataclass
class NearMiss:
    track_id: int
    class_name: str
    min_abs_distance_px: float
    frames_seen: int


class RecallChecker:
    """Flags tracks that came close to the counting line but were never
    confirmed as crossing it - a cheap proxy for missed counts (track
    lost right at the boundary) without needing hand-labeled ground
    truth. A flagged track isn't necessarily a miss - it may be a real
    vehicle that slowed, turned off, or was still near the line when
    the video ended - so this is a shortlist to spot-check, not a
    guaranteed error count.
    """

    def __init__(self, threshold_px: float = 120):
        self.threshold_px = threshold_px
        self._min_distance: dict[int, float] = {}
        self._class_name: dict[int, str] = {}
        self._frames_seen: dict[int, int] = {}

    def observe(self, tracked_objects, counter) -> None:
        for obj in tracked_objects:
            dist = abs(counter.distance_to_line(obj.xyxy))
            self._min_distance[obj.track_id] = min(dist, self._min_distance.get(obj.track_id, float("inf")))
            self._class_name[obj.track_id] = obj.class_name
            self._frames_seen[obj.track_id] = self._frames_seen.get(obj.track_id, 0) + 1

    def near_misses(self, counter) -> list[NearMiss]:
        flagged = []
        for track_id, min_dist in self._min_distance.items():
            if min_dist <= self.threshold_px and not counter.is_counted(track_id):
                flagged.append(
                    NearMiss(track_id, self._class_name[track_id], round(min_dist, 1), self._frames_seen[track_id])
                )
        return sorted(flagged, key=lambda n: n.min_abs_distance_px)
