# Project Objectives

1. Calibrate the camera using at least 20 checkerboard images.
2. Report camera matrix, distortion coefficients, and reprojection error.
3. Build a segmentation dataset containing at least 70 original object images.
4. Train and evaluate an instance segmentation model using IoU, mAP@0.5, mAP@0.5:0.95, precision, and recall.
5. Convert segmented object dimensions from pixels to millimeters using a detected reference.
6. Compare estimated dimensions against physical measurements for at least 10 object instances.
7. Report MAE and mean percentage error.
8. Compare measurement errors using raw and undistorted images.
9. Detect at least one unreliable input condition and refuse to report a measurement when it occurs.