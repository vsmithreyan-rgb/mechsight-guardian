import cv2
import os

from vision.feature_extractor import extract_visual_features


def analyse_folder(folder_path, label):
    results = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not load: {image_path}")
                continue

            features = extract_visual_features(image)

            results.append({
                "filename": filename,
                "label": label,
                **features
            })

    return results


leak_results = analyse_folder("data/leak", "LEAK")
normal_results = analyse_folder("data/normal", "NORMAL")

all_results = leak_results + normal_results


print("\nMECHSIGHT FEATURE COMPARISON")
print("=" * 170)

print(
    f"{'IMAGE':<18}"
    f"{'LABEL':<9}"
    f"{'EDGE':<10}"
    f"{'SAT':<10}"
    f"{'BRIGHT':<10}"
    f"{'CONTRAST':<11}"
    f"{'DARK':<10}"
    f"{'HIGH SAT':<11}"
    f"{'TEXTURE':<14}"
    f"{'LOW DARK':<11}"
    f"{'CENTRE':<10}"
)

print("-" * 170)

for result in all_results:
    print(
        f"{result['filename']:<18}"
        f"{result['label']:<9}"
        f"{result['edge_density']:<10.4f}"
        f"{result['mean_saturation']:<10.2f}"
        f"{result['mean_brightness']:<10.2f}"
        f"{result['contrast']:<11.2f}"
        f"{result['dark_region_ratio']:<10.4f}"
        f"{result['saturated_region_ratio']:<11.4f}"
        f"{result['texture_variance']:<14.2f}"
        f"{result['lower_dark_ratio']:<11.4f}"
        f"{result['centre_contrast']:<10.2f}"
    )

print("=" * 170)
print(f"Total images tested: {len(all_results)}")
