import cv2
import numpy as np


def extract_visual_features(image):
    # Resize for consistent feature extraction
    image = cv2.resize(image, (500, 500))

    # Convert to useful colour spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]

    # -------------------------------------------------
    # 1. EDGE DENSITY
    # -------------------------------------------------
    edges = cv2.Canny(gray, 50, 150)

    edge_density = (
        np.count_nonzero(edges) / edges.size
    )

    # -------------------------------------------------
    # 2. GLOBAL BRIGHTNESS / SATURATION
    # -------------------------------------------------
    mean_saturation = np.mean(saturation)
    mean_brightness = np.mean(brightness)

    # -------------------------------------------------
    # 3. IMAGE CONTRAST
    # -------------------------------------------------
    contrast = np.std(gray)

    # -------------------------------------------------
    # 4. DARK REGION RATIO
    # Useful for dark oil / wet areas
    # -------------------------------------------------
    dark_mask = gray < 70

    dark_region_ratio = (
        np.count_nonzero(dark_mask)
        / dark_mask.size
    )

    # -------------------------------------------------
    # 5. HIGH SATURATION REGION RATIO
    # Useful for coloured fluids / stains
    # -------------------------------------------------
    saturated_mask = saturation > 100

    saturated_region_ratio = (
        np.count_nonzero(saturated_mask)
        / saturated_mask.size
    )

    # -------------------------------------------------
    # 6. TEXTURE MEASUREMENT
    # Laplacian variance measures local texture/detail
    # -------------------------------------------------
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    texture_variance = laplacian.var()

    # -------------------------------------------------
    # 7. LOWER IMAGE DARKNESS
    # Puddles often occur lower in the frame.
    # This is only evidence, NOT a leak rule.
    # -------------------------------------------------
    height = gray.shape[0]

    lower_region = gray[
        int(height * 0.60):height,
        :
    ]

    lower_dark_ratio = (
        np.count_nonzero(lower_region < 70)
        / lower_region.size
    )

    # -------------------------------------------------
    # 8. CENTRE REGION CONTRAST
    # Useful for defects around inspected equipment
    # -------------------------------------------------
    h, w = gray.shape

    centre_region = gray[
        int(h * 0.25):int(h * 0.75),
        int(w * 0.25):int(w * 0.75)
    ]

    centre_contrast = np.std(centre_region)

    # -------------------------------------------------
    # RETURN FEATURES
    # -------------------------------------------------
    features = {
        "edge_density": float(edge_density),
        "mean_saturation": float(mean_saturation),
        "mean_brightness": float(mean_brightness),
        "contrast": float(contrast),
        "dark_region_ratio": float(dark_region_ratio),
        "saturated_region_ratio": float(
            saturated_region_ratio
        ),
        "texture_variance": float(
            texture_variance
        ),
        "lower_dark_ratio": float(
            lower_dark_ratio
        ),
        "centre_contrast": float(
            centre_contrast
        )
    }

    return features