import cv2


def preprocess_image(image):
    resized = cv2.resize(image, (500, 500))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    return blurred
