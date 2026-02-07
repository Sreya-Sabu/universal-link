# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib
# import time

# model = joblib.load('asl_model.pkl')
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)

# cap = cv2.VideoCapture(0)

# sentence = []
# current_word = ""
# word_confidence_counter = 0
# last_hand_time = time.time()

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv2.flip(frame, 1)
#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     if results.multi_hand_landmarks:
#         last_hand_time = time.time()
#         landmarks = []
#         for lm in results.multi_hand_landmarks[0].landmark:
#             landmarks.extend([lm.x, lm.y, lm.z])
        
#         # Predict
#         pred = model.predict([landmarks])[0]
#         conf = np.max(model.predict_proba([landmarks]))

#         if conf > 0.7:
#             if pred == current_word:
#                 word_confidence_counter += 1
#             else:
#                 current_word = pred
#                 word_confidence_counter = 0
            
#             # If we held the sign for 20 consecutive frames, lock it in
#             if word_confidence_counter == 20:
#                 if len(sentence) == 0 or sentence[-1] != pred:
#                     sentence.append(pred)
#                 word_confidence_counter = 0
#     else:
#         # If no hand for 2 seconds, assume end of word/add space logic
#         if time.time() - last_hand_time > 2.0 and len(sentence) > 0 and sentence[-1] != " ":
#             # Optionally add a space or just reset
#             pass

#     # UI Overlay
#     display_text = " ".join(sentence)
#     cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
#     cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#     cv2.putText(frame, f"Detecting: {current_word}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow('ASL Sentence Translator', frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'): break
#     if key == ord('c'): sentence = [] # Press 'c' to clear sentence

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import joblib

# Config
MODEL_PATH = 'asl_model.pkl'
THRESHOLD = 10  # Very snappy lock-in
CONF_MIN = 0.75 # Confidence filter

model = joblib.load(MODEL_PATH)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sentence = []
current_word = ""
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        hand_lms = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        # --- CENTERING LOGIC (Must match collector) ---
        wrist = hand_lms.landmark[0]
        landmarks = []
        for lm in hand_lms.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        
        # Predict
        probs = model.predict_proba([landmarks])
        conf = np.max(probs)
        pred = model.predict([landmarks])[0]

        if conf > CONF_MIN:
            if pred == current_word:
                counter += 1
            else:
                current_word = pred
                counter = 0
            
            if counter == THRESHOLD:
                if not sentence or sentence[-1] != pred:
                    sentence.append(pred)
                counter = 0
    else:
        current_word = ""
        counter = 0

    # --- UI ---
    # Top Sentence Bar
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)
    cv2.putText(frame, " ".join(sentence).upper(), (20, 40), 1, 2, (0, 255, 255), 2)
    
    # Bottom Status
    if current_word:
        cv2.putText(frame, f"SIGNING: {current_word.upper()}", (20, h-20), 1, 1.5, (0, 255, 0), 2)

    cv2.imshow('ASL Universal Link', frame)
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): sentence = []

cap.release()
cv2.destroyAllWindows()