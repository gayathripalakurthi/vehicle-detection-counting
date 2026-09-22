"""Single-operating-point detection metrics (precision/recall/F1 at a fixed
confidence threshold, via greedy IoU matching) - the standard way to report
"how good is this detector at the threshold I'm actually going to run it
at", as distinct from full mAP which sweeps confidence.

Matching is by class *name*, not class id, so a COCO-pretrained model
(ids like 2=car) and a model fine-tuned with its own id scheme (e.g.
1=car) can be compared fairly on the same footing.
"""

from dataclasses import dataclass


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class ClassCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


class DetectionEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.counts: dict[str, ClassCounts] = {}

    def update(
        self,
        predictions: list[tuple[str, float, tuple[float, float, float, float]]],
        ground_truths: list[tuple[str, tuple[float, float, float, float]]],
    ) -> None:
        preds_sorted = sorted(predictions, key=lambda p: -p[1])
        matched_gt = [False] * len(ground_truths)

        for class_name, _conf, box in preds_sorted:
            counts = self.counts.setdefault(class_name, ClassCounts())
            best_iou, best_idx = 0.0, -1
            for i, (gt_name, gt_box) in enumerate(ground_truths):
                if matched_gt[i] or gt_name != class_name:
                    continue
                cur_iou = iou(box, gt_box)
                if cur_iou > best_iou:
                    best_iou, best_idx = cur_iou, i
            if best_iou >= self.iou_threshold:
                matched_gt[best_idx] = True
                counts.tp += 1
            else:
                counts.fp += 1

        for i, (gt_name, _box) in enumerate(ground_truths):
            if not matched_gt[i]:
                self.counts.setdefault(gt_name, ClassCounts()).fn += 1

    def report(self) -> dict:
        out = {}
        total = ClassCounts()
        for name, c in sorted(self.counts.items()):
            out[name] = {
                "precision": round(c.precision(), 4),
                "recall": round(c.recall(), 4),
                "f1": round(c.f1(), 4),
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
            }
            total.tp += c.tp
            total.fp += c.fp
            total.fn += c.fn
        out["_overall"] = {
            "precision": round(total.precision(), 4),
            "recall": round(total.recall(), 4),
            "f1": round(total.f1(), 4),
            "tp": total.tp,
            "fp": total.fp,
            "fn": total.fn,
        }
        return out
