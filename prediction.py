#nlp using mapping


import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# ===================== CONFIG =====================
FACE_LMS = [33,133,362,263,159,145,386,374,10,152,61,291,2]
POSE_LMS = [11,12,13,14,15,16]
CONF_THRESHOLD = 0.5
BUFFER_LEN = 5
# Minimum times the top label must appear in the buffer to be accepted
MIN_COUNT = max(1, BUFFER_LEN // 2 + 1)
# ==================================================

# Load trained sign language model
model = joblib.load("sign_language_model.pkl")

# Mediapipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5
)

# Mapping raw sentences to grammatically correct sentences
phrase_mapping = {
    "i sorry": "I am sorry",
    "i hungry": "I am hungry",
    "i fine": "I am Fine",
    "where you": "Where Are You",
    "i eat": "I am eating",
    "how you": "How are you",
    "i happy":"I am Happy",
    "i tired":"I am Tired",
    "who you":"Who are You",
    "what you name":"What is your name"
    # Add more phrases here as needed
}

# Controls
print("==== Controls ====")
print("s = start sentence recording")
print("e = end sentence recording and show result")
print("q = quit")
print("==================")

# Buffers
word_buffer = deque(maxlen=BUFFER_LEN)
sentence_words = []
recording = False

def extract_features(results):
    """Extracts landmarks in same order as dataset collection"""
    features = []

    # Pose
    if results.pose_landmarks:
        for i in POSE_LMS:
            lm = results.pose_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0.0] * (len(POSE_LMS)*4))

    # Face
    if results.face_landmarks:
        for i in FACE_LMS:
            lm = results.face_landmarks.landmark[i]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * (len(FACE_LMS)*3))

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0]*63)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0]*63)

    return np.array(features).reshape(1, -1)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    pred = "NULL"
    pred_prob = 0.0

    if recording:
        if results.left_hand_landmarks or results.right_hand_landmarks:
            X = extract_features(results)

            # Use predict_proba when available to filter low-confidence predictions
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                idx = np.argmax(probs, axis=1)[0]
                pred = model.classes_[idx]
                pred_prob = float(probs[0][idx])
            else:
                pred = model.predict(X)[0]
                pred_prob = 1.0

            # Only consider predictions above confidence threshold
            if pred != "NULL" and pred_prob >= CONF_THRESHOLD:
                word_buffer.append(pred)

                # Stability check: accept if the most common label appears at least MIN_COUNT times
                if len(word_buffer) == BUFFER_LEN:
                    top_label, top_count = Counter(word_buffer).most_common(1)[0]
                    if top_label != "NULL" and top_count >= MIN_COUNT:
                        if len(sentence_words) == 0 or sentence_words[-1] != top_label:
                            sentence_words.append(top_label)
                            print(f" Added word: {top_label} (count={top_count}, prob={pred_prob:.2f})")
            else:
                # If prediction is weak, push a placeholder to preserve temporal smoothing window
                word_buffer.append("NULL")
        else:
            word_buffer.clear()
            pred = "NULL"

    # show probability on frame if available
    prob_text = f" ({pred_prob:.2f})" if pred_prob else ""

    if recording:
        cv2.putText(frame, f"Recording... Words: {' '.join(sentence_words)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "Stopped (press 's' to start)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame, f"Word: {pred}{prob_text}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Sign Language Sequential Prediction", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        sentence_words = []
        recording = True
        print("Recording started... perform your sentence")

    elif key == ord('e'):
        recording = False
        raw_sentence = " ".join(sentence_words).lower()
        print(" Recording stopped.")
        print(" Raw sentence:", raw_sentence)

        # Apply mapping for grammar correction
        corrected_sentence = phrase_mapping.get(raw_sentence, raw_sentence)
        print(" Corrected sentence:", corrected_sentence)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()