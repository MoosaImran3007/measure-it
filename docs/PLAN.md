# Project Plan — Measure It

## 1. Project Goal
Build a computer vision measurement pipeline that calibrates a camera, segments an object, and estimates its physical dimensions in millimeters.

## 2. Implementation Phases
1. Create and verify the project environment.
2. Capture at least 20 checkerboard calibration images.
3. Calibrate the camera and save its matrix and distortion coefficients.
4. Collect and annotate at least 70 object images.
5. Create and commit a reproducible 70/20/10 train-validation-test split.
6. Train and evaluate an instance segmentation model.
7. Implement image undistortion, segmentation, reference detection, and measurement.
8. Evaluate at least 10 physical object instances against ruler or caliper measurements.
9. Test a failure condition and implement a reliability check.
10. Prepare documentation, demo, and submission.

## 3. Calibration Setup
- Checkerboard inner corners: 7 × 9
- Square size: 24 mm
- Minimum calibration images: 20
- Keep camera resolution, zoom, and focus fixed.

## 4. Risks and Mitigations
- Board bending: mount it flat on a rigid backing.
- Blur or glare: use sharp, evenly lit images.
- Poor calibration coverage: vary board position, angle, and distance.
- Unreliable measurements: define and test a refusal condition.

## 5. Deliverables
- Calibration parameters and reprojection error
- Dataset and committed split list
- Trained segmentation model and evaluation metrics
- Measurement pipeline and overlay output
- Physical measurement comparison table, MAE, and mean percentage error
- Documentation, demo, and submission PRs