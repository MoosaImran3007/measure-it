import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# Board has 8 x 10 squares -> 7 x 9 inner corners
BOARD_SIZE = (7, 9)
SQUARE_SIZE_MM = 24.0

IMAGE_DIR = Path("data/calibration")
OUTPUT_DIR = Path("calibration")
OUTPUT_FILE = OUTPUT_DIR / "camera_calibration.npz"


def detect_checkerboard(gray):
    """Detect the checkerboard using both orientations and available methods."""
    patterns = [BOARD_SIZE, (BOARD_SIZE[1], BOARD_SIZE[0])]

    # Try the more robust sector-based detector first.
    if hasattr(cv2, "findChessboardCornersSB"):
        for pattern in patterns:
            found, corners = cv2.findChessboardCornersSB(gray, pattern)
            if found:
                return pattern, corners

    # Fall back to the classic detector.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    for pattern in patterns:
        found, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                ),
            )
            return pattern, corners

    return None, None


def make_object_points(pattern):
    """Create checkerboard corner coordinates in millimeters."""
    points = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float32)
    points[:, :2] = (
        np.mgrid[0:pattern[0], 0:pattern[1]]
        .T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )
    return points


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = sorted(
        path for ext in extensions for path in IMAGE_DIR.glob(ext)
    )

    if len(image_paths) < 20:
        print(f"Need at least 20 calibration images; found {len(image_paths)}.")
        print(f"Place images in: {IMAGE_DIR.resolve()}")
        return

    # Read images and identify the most common resolution.
    readable_images = []
    sizes = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        size = (image.shape[1], image.shape[0])
        readable_images.append((image_path, image, size))
        sizes.append(size)

    if not readable_images:
        print("No readable images found.")
        return

    image_size = Counter(sizes).most_common(1)[0][0]
    print(f"Using most common image resolution: {image_size}")

    all_object_points = []
    all_image_points = []

    for image_path, image, current_size in readable_images:
        if current_size != image_size:
            print(
                f"Skipping {image_path.name}: size {current_size} "
                f"differs from selected size {image_size}."
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pattern, corners = detect_checkerboard(gray)

        if pattern is None:
            print(f"Could not detect checkerboard: {image_path.name}")
            continue

        all_object_points.append(make_object_points(pattern))
        all_image_points.append(corners)
        print(f"Detected {pattern[0]}x{pattern[1]} corners: {image_path.name}")

    detected = len(all_object_points)
    print(f"\nDetected checkerboards in {detected} images at {image_size}.")

    if detected < 20:
        print("Need at least 20 successfully detected images at the same resolution.")
        return

    rms, camera_matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        reprojection_error=rms,
    )

    print("\nCalibration complete.")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", distortion)
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print(f"Saved calibration to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()