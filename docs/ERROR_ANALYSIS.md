# RoadGuard AI error analysis

This report uses two distinct evidence sources:

- Ultralytics evaluation over all 796 images in the frozen held-out test split.
- A manually inspectable 24-image sample drawn from that same split at confidence 0.25
  and class-aware matching IoU 0.50. This small sample supports examples and failure-mode
  inspection; it does not replace the full evaluator.

## Full-test class behavior

The selected YOLO26n checkpoint achieved overall test mAP@0.50:0.95 of 0.1924.
Its weakest category was D40 Pothole: precision 0.3758, recall 0.2197,
AP@0.50 0.2161, and AP@0.50:0.95 0.0861. D20 Alligator Crack had the highest
class AP@0.50:0.95 (0.2577), while D00 had the highest recall (0.5208). The
committed confusion matrix, PR curve, and F1 curve come directly from the same
held-out evaluator.

## Inspectable sample

The 24-image sample contained 42 ground-truth objects and produced 20 detections.
Greedy same-class matching at IoU 0.50 found 11 true positives, 9 false positives,
and 31 false negatives: sample precision 0.5500, sample recall 0.2619, and mean
IoU 0.7211 across matched boxes. D20 and D40 have only three and one ground-truth
instances in this sample, so their sample-level match counts are not used as stable
class estimates.

Concrete inspected cases:

- `China_Drone_000166.jpg`: two matched D00 detections; a genuine positive example.
- `Czech_000007.jpg`: the source label is empty, but the detector returned two D00
  boxes. Visual inspection shows tram rails, lane markings, and long linear road
  features—a plausible false-positive pattern for crack-like geometry.
- `Norway_000165.jpg`: thirteen labeled D00/D10 instances and no detections. The
  defects are numerous, narrow, and partly mixed with repaired pavement, glare, and
  tree shadows; this is a clear dense-small-crack failure case.
- `United_States_000097.jpg`: four detections, with two same-class matches; it shows
  both successful localization and duplicate/misaligned candidates in one scene.

## Known limitations and next experiments

- Pothole recall is currently too low for safety-critical or unattended inspection.
- Thin, dense cracks and damage under shadow or high-contrast illumination are often
  missed.
- Long road markings, tram rails, repair seams, and pavement texture can resemble
  longitudinal cracks.
- The six-country subset improves diversity but does not prove route-level or global
  geographic generalization.
- A stronger follow-up should increase D40 coverage, use hard-negative mining for
  road markings and repair seams, stratify by country/camera domain, and evaluate
  confidence calibration. Test data must remain excluded from those choices.

Machine-readable per-image evidence is in
`artifacts/metrics/representative_error_analysis.json`.
