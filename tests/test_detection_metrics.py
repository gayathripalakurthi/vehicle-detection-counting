from src.evaluation.detection_metrics import DetectionEvaluator, iou


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_perfect_prediction_counts_as_true_positive():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    box = (10.0, 10.0, 50.0, 50.0)
    evaluator.update(predictions=[("car", 0.9, box)], ground_truths=[("car", box)])
    report = evaluator.report()
    assert report["car"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0}


def test_missed_detection_counts_as_false_negative():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    evaluator.update(predictions=[], ground_truths=[("car", (10.0, 10.0, 50.0, 50.0))])
    report = evaluator.report()
    assert report["car"]["fn"] == 1
    assert report["car"]["recall"] == 0.0


def test_spurious_prediction_counts_as_false_positive():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    evaluator.update(predictions=[("car", 0.8, (10.0, 10.0, 50.0, 50.0))], ground_truths=[])
    report = evaluator.report()
    assert report["car"]["fp"] == 1
    assert report["car"]["precision"] == 0.0


def test_low_overlap_prediction_is_false_positive_not_true_positive():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    # boxes barely overlap - IoU well under 0.5
    evaluator.update(
        predictions=[("car", 0.8, (0.0, 0.0, 10.0, 10.0))],
        ground_truths=[("car", (8.0, 8.0, 18.0, 18.0))],
    )
    report = evaluator.report()
    assert report["car"]["tp"] == 0
    assert report["car"]["fp"] == 1
    assert report["car"]["fn"] == 1  # the real car was never matched either


def test_wrong_class_never_matches_even_with_perfect_overlap():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    box = (10.0, 10.0, 50.0, 50.0)
    evaluator.update(predictions=[("truck", 0.9, box)], ground_truths=[("car", box)])
    report = evaluator.report()
    assert report["truck"]["tp"] == 0 and report["truck"]["fp"] == 1
    assert report["car"]["fn"] == 1


def test_duplicate_predictions_on_one_ground_truth_only_one_is_a_true_positive():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    box = (10.0, 10.0, 50.0, 50.0)
    evaluator.update(
        predictions=[("car", 0.95, box), ("car", 0.60, box)],  # two predictions, one real car
        ground_truths=[("car", box)],
    )
    report = evaluator.report()
    assert report["car"]["tp"] == 1
    assert report["car"]["fp"] == 1  # the lower-confidence duplicate


def test_overall_aggregates_across_classes():
    evaluator = DetectionEvaluator(iou_threshold=0.5)
    box = (0.0, 0.0, 10.0, 10.0)
    evaluator.update(predictions=[("car", 0.9, box)], ground_truths=[("car", box)])
    evaluator.update(predictions=[], ground_truths=[("bus", (20.0, 20.0, 30.0, 30.0))])
    report = evaluator.report()
    assert report["_overall"]["tp"] == 1
    assert report["_overall"]["fn"] == 1
