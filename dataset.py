#Code for dataset collection

import cv2
import mediapipe as mp
import csv
import os

# ===================== CONFIG =====================
# Selected landmark indices
FACE_LMS = [
    33, 133,       # left eye corners
    362, 263,      # right eye corners
    159, 145,      # left eyebrow (upper/lower mid)
    386, 374,      # right eyebrow (upper/lower mid)
    10, 152,       # forehead top & chin (face outline)
    61, 291,       # mouth corners
    2              # nose tip
]
POSE_LMS = [11, 12, 13, 14, 15, 16]  # shoulders + elbows
# ==================================================

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Open CSV file
csv_file = open('dataset/sign_data.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# Write header if file is empty
if os.stat('dataset/sign_data.csv').st_size == 0:
    header = ["label"]

    # Pose
    for i in POSE_LMS:
        header += [f"pose_x{i}", f"pose_y{i}", f"pose_z{i}", f"pose_vis{i}"]

    # Face (reduced set)
    for i in FACE_LMS:
        header += [f"face_x{i}", f"face_y{i}", f"face_z{i}"]

    # Hands
    for i in range(21):
        header += [f"left_hand_x{i}", f"left_hand_y{i}", f"left_hand_z{i}"]
    for i in range(21):
        header += [f"right_hand_x{i}", f"right_hand_y{i}", f"right_hand_z{i}"]

    csv_writer.writerow(header)

# Load webcam
cap = cv2.VideoCapture(0)

# Labels will be added by user
labels = []
label = ""
label_counts = {}
frame_count = 0
recording = False  # flag to control recording

print("==== Controls ====")
print("s = start recording")
print("e = stop recording")
print("m = enter/change label manually")
print("v = view current label stats")
print("q = quit")
print("==================")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # ---------- Draw Landmarks ----------
    # Pose
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Hands
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Face (reduced points only, color coded)
    if results.face_landmarks:
        h, w, c = frame.shape
        for i in FACE_LMS:
            lm = results.face_landmarks.landmark[i]
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Color groups
            if i in [33, 133, 362, 263]:  # eyes
                color = (255, 0, 0)  # blue
            elif i in [159, 145, 386, 374]:  # eyebrows
                color = (0, 255, 0)  # green
            elif i in [61, 291]:  # mouth corners
                color = (0, 0, 255)  # red
            elif i in [10, 152]:  # forehead + chin
                color = (255, 255, 0)  # cyan
            elif i == 2:  # nose tip
                color = (255, 0, 255)  # magenta
            else:
                color = (0, 255, 255)  # yellow

            cv2.circle(frame, (cx, cy), 3, color, -1)

    # ---------- Save Landmarks ----------
    if recording and label != "":
        row = [label]

        # Pose
        if results.pose_landmarks:
            for i in POSE_LMS:
                lm = results.pose_landmarks.landmark[i]
                row += [lm.x, lm.y, lm.z, lm.visibility]
        else:
            row += [0.0] * (len(POSE_LMS) * 4)

        # Face
        if results.face_landmarks:
            for i in FACE_LMS:
                lm = results.face_landmarks.landmark[i]
                row += [lm.x, lm.y, lm.z]
        else:
            row += [0.0] * (len(FACE_LMS) * 3)

        # Left Hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
        else:
            row += [0.0] * (21 * 3)

        # Right Hand
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
        else:
            row += [0.0] * (21 * 3)

        csv_writer.writerow(row)
        frame_count += 1
        label_counts[label] += 1
        print(f"Saved frame {frame_count} for {label}")

    # ---------- Status Display ----------
    status = "Recording..." if recording else "Stopped"
    cv2.putText(frame,
                f"Label: {label if label else 'None'} | {status} | Samples: {label_counts.get(label,0)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if recording else (0, 255, 0), 2)

    cv2.imshow("Mirror View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):   # Start recording
        if label == "":
            print("⚠ Please set a label first using 'm'")
        else:
            recording = True
            print(f"Recording started for label: {label}")
    elif key == ord('e'): # Stop recording
        recording = False
        print("Recording stopped.")
    elif key == ord('m'): # Manual edit
        recording = False
        new_label = input("Enter new label: ").strip()
        if new_label != "":
            label = new_label
            if label not in labels:
                labels.append(label)
                label_counts[label] = 0
            print(f"Label changed to: {label}")
    elif key == ord('v'): # View stats
        print("\n=== Label Stats ===")
        for lbl, count in label_counts.items():
            print(f"{lbl}: {count} samples")
        print("===================\n")
    elif key == ord('q'): # Quit
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()