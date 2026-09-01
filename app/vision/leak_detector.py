import cv2
import numpy as np
from vision.preprocessing import preprocess_image


def detect_leak(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image.")
        return

    print("Image loaded successfully!")

    processed_image = preprocess_image(image)

    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

    lower_color = np.array([5, 100, 80])
    upper_color = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    total_leak_area = 0
    detected_regions = 0

    image_height = processed_image.shape[0]
    image_width = processed_image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 1500:
            x, y, w, h = cv2.boundingRect(contour)

            bounding_area = w * h
            fill_ratio = area / bounding_area

            aspect_ratio = w / h if h != 0 else 0

            region_center_y = y + (h / 2)

            is_lower_region = (
                region_center_y > image_height * 0.35
            )

            has_good_fill = fill_ratio > 0.25

            has_reasonable_shape = (
                0.2 < aspect_ratio < 5.0
            )

            if (
                has_good_fill
                and has_reasonable_shape
                and is_lower_region
            ):
                detected_regions += 1
                total_leak_area += area

                cv2.rectangle(
                    processed_image,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    processed_image,
                    f"Leak Region {detected_regions}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    image_area = image_width * image_height

    leak_percentage = (
        total_leak_area / image_area
    ) * 100

    print(f"Detected regions: {detected_regions}")
    print(f"Detected leak area: {total_leak_area:.0f} pixels")
    print(f"Leak coverage: {leak_percentage:.2f}%")

    cv2.putText(
        processed_image,
        f"Leak Coverage: {leak_percentage:.2f}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.namedWindow(
        "MechSight - Detection",
        cv2.WINDOW_NORMAL
    )

    cv2.namedWindow(
        "MechSight - Leak Mask",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "MechSight - Detection",
        600,
        600
    )

    cv2.resizeWindow(
        "MechSight - Leak Mask",
        600,
        600
    )

    cv2.moveWindow(
        "MechSight - Detection",
        50,
        50
    )

    cv2.moveWindow(
        "MechSight - Leak Mask",
        700,
        50
    )

    cv2.imshow(
        "MechSight - Detection",
        processed_image
    )

    cv2.imshow(
        "MechSight - Leak Mask",
        mask
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()