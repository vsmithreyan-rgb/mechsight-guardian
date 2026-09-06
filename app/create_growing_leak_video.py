import cv2
import numpy as np
import os
import random

OUTPUT_PATH = "data/test/growing_leak_video.mp4"

os.makedirs("data/test", exist_ok=True)

width = 1280
height = 720
fps = 15
total_frames = 150  # 10 seconds

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

for frame_number in range(total_frames):

    # Industrial-looking background
    frame = np.full((height, width, 3), (55, 60, 65), dtype=np.uint8)

    # -------------------------
    # PIPE
    # -------------------------
    pipe_y = 260

    cv2.rectangle(
        frame,
        (100, pipe_y),
        (1180, pipe_y + 150),
        (125, 125, 125),
        -1
    )

    # Pipe highlights/shadows
    cv2.line(frame, (100, pipe_y + 20), (1180, pipe_y + 20),
             (175, 175, 175), 5)

    cv2.line(frame, (100, pipe_y + 130), (1180, pipe_y + 130),
             (80, 80, 80), 8)

    # -------------------------
    # PIPE JOINT
    # -------------------------
    joint_x = 650

    cv2.rectangle(
        frame,
        (joint_x - 45, pipe_y - 15),
        (joint_x + 45, pipe_y + 165),
        (90, 90, 90),
        -1
    )

    # Rust around leak
    cv2.circle(
        frame,
        (joint_x + 45, pipe_y + 80),
        38,
        (40, 75, 110),
        -1
    )

    # -------------------------
    # GROWING LEAK
    # -------------------------
    progress = frame_number / (total_frames - 1)

    # Leak starts small and grows
    stream_length = int(70 + progress * 330)
    stream_width = int(5 + progress * 22)

    leak_x = joint_x + 45
    leak_y = pipe_y + 80

    # Main water stream
    points = []

    for x in range(leak_x, leak_x + stream_length, 5):

        distance = x - leak_x

        # Downward curve
        y = leak_y + int(0.0018 * distance ** 2)

        # Moving turbulence
        wave = int(
            np.sin(frame_number * 0.35 + distance * 0.08) * 8
        )

        points.append((x, y + wave))

    for i in range(len(points) - 1):

        thickness = max(
            2,
            int(stream_width * (1 - i / (len(points) * 1.5)))
        )

        cv2.line(
            frame,
            points[i],
            points[i + 1],
            (230, 220, 190),
            thickness
        )

    # -------------------------
    # MOVING WATER DROPLETS
    # -------------------------
    random.seed(frame_number)

    droplet_count = int(5 + progress * 35)

    for _ in range(droplet_count):

        dx = random.randint(20, max(21, stream_length))
        dy = int(0.0018 * dx ** 2)

        droplet_x = leak_x + dx
        droplet_y = (
            leak_y
            + dy
            + random.randint(-35, 35)
        )

        if 0 <= droplet_x < width and 0 <= droplet_y < height:

            radius = random.randint(2, 5)

            cv2.circle(
                frame,
                (droplet_x, droplet_y),
                radius,
                (240, 230, 200),
                -1
            )

    # -------------------------
    # LABEL
    # -------------------------
    cv2.putText(
        frame,
        "SYNTHETIC LEAK TEST - DEVELOPMENT ONLY",
        (35, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    writer.write(frame)

writer.release()

print(f"Synthetic moving leak video created: {OUTPUT_PATH}")