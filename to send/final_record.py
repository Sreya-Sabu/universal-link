# import cv2
# import csv
# import mediapipe as mp
# import numpy as np

# # Setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
# mp_drawing = mp.solutions.drawing_utils

# TARGET_COUNT = 200 

# while True:
#     label = input("\nEnter Word (or 'exit'): ").strip().lower()
#     if label == 'exit': break
    
#     cap = cv2.VideoCapture(0)
#     print(f"Prepare to sign: {label}")
#     cv2.waitKey(2000)
    
#     count = 0
#     with open('my_data.csv', 'a', newline='') as f:
#         writer = csv.writer(f)
#         while count < TARGET_COUNT:
#             ret, frame = cap.read()
#             if not ret: break
#             frame = cv2.flip(frame, 1)
#             results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
#             if results.multi_hand_landmarks:
#                 hand_lms = results.multi_hand_landmarks[0]
                
#                 # --- CENTERING LOGIC ---
#                 # Wrist is landmark 0
#                 wrist = hand_lms.landmark[0]
#                 landmarks = []
#                 for lm in hand_lms.landmark:
#                     # Subtract wrist from every point
#                     landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
#                 writer.writerow([label] + landmarks)
#                 count += 1
#                 mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
#             cv2.putText(frame, f"Rec: {label} ({count}/{TARGET_COUNT})", (10, 50), 1, 2, (0,255,0), 2)
#             cv2.imshow('Recording', frame)
#             if cv2.waitKey(1) == ord('q'): break
#     cap.release()
#     cv2.destroyAllWindows()

import cv2
import csv
import mediapipe as mp
import numpy as np

# --- 1. SET TO 2 HANDS ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2) 
mp_drawing = mp.solutions.drawing_utils

TARGET_COUNT = 200 

while True:
    label = input("\nEnter Word (or 'exit'): ").strip().lower()
    if label == 'exit': break
    
    cap = cv2.VideoCapture(0)
    print(f"Prepare to sign: {label}")
    cv2.waitKey(2000)
    
    count = 0
    with open('my_data.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        while count < TARGET_COUNT:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # --- 2. CREATE 126 FEATURE PLACEHOLDER ---
            # This ensures we always have (63 * 2) = 126 coordinates
            row_data = np.zeros(126) 
            
            if results.multi_hand_landmarks:
                for i, hand_lms in enumerate(results.multi_hand_landmarks):
                    if i < 2: # Only take first two hands
                        wrist = hand_lms.landmark[0]
                        hand_coords = []
                        for lm in hand_lms.landmark:
                            hand_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                        
                        # Fill the specific 63-slot segment for this hand
                        row_data[i*63 : (i+1)*63] = hand_coords
                        mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # --- 3. SAVE THE FULL 126 FEATURES ---
                writer.writerow([label] + row_data.tolist())
                count += 1
            
            cv2.putText(frame, f"Rec: {label} ({count}/{TARGET_COUNT})", (10, 50), 1, 2, (0,255,0), 2)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()