# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib

# # --- CONFIG ---
# MODEL_PATH = 'asl_model.pkl'
# THRESHOLD = 8   # Number of frames to "lock in" a word
# CONF_MIN = 0.75 # Ignore predictions with low confidence

# # Load the trained "Two-Hand" brain
# try:
#     model = joblib.load(MODEL_PATH)
# except:
#     print("Error: Could not find asl_model.pkl. Run train_model.py first.")
#     exit()

# # Setup Mediapipe for 2 Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)
# sentence = []
# current_word = ""
# counter = 0

# print("Predictor started. Press 'C' to clear sentence, 'Q' to quit.")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
    
#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     # 1. Prepare an empty 126-feature row (All zeros)
#     row_data = np.zeros(126)
#     prediction_made = False

#     if results.multi_hand_landmarks:
#         # 2. Extract and Centralize landmarks for up to 2 hands
#         for i, hand_lms in enumerate(results.multi_hand_landmarks):
#             if i < 2:
#                 wrist = hand_lms.landmark[0]
#                 hand_coords = []
#                 for lm in hand_lms.landmark:
#                     # Same Centering Logic: lm - wrist
#                     hand_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
#                 # Fill the correct section of the 126-feature row
#                 row_data[i*63 : (i+1)*63] = hand_coords
#                 mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
#         # 3. Prediction
#         probs = model.predict_proba([row_data])
#         conf = np.max(probs)
#         pred = model.predict([row_data])[0]

#         if conf > CONF_MIN:
#             prediction_made = True
#             if pred == current_word:
#                 counter += 1
#             else:
#                 current_word = pred
#                 counter = 0
            
#             # Lock-in word
#             if counter == THRESHOLD:
#                 if pred != "none": # Don't add 'none' to the actual sentence
#                     if not sentence or sentence[-1] != pred:
#                         sentence.append(pred)
#                 counter = 0
#     else:
#         # No hands visible
#         current_word = ""
#         counter = 0

#     # --- UI ---
#     # Top Sentence Bar
#     cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)
#     display_sentence = " ".join(sentence).upper()
#     cv2.putText(frame, display_sentence, (20, 45), 1, 2, (0, 255, 255), 2)
    
#     # Bottom Status (Feedback on current gesture)
#     if prediction_made and current_word != "none":
#         cv2.putText(frame, f"SIGN: {current_word.upper()}", (20, h-20), 1, 1.5, (0, 255, 0), 2)

#     cv2.imshow('ASL Two-Hand Predictor', frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'): break
#     if key == ord('c'): sentence = []

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import mediapipe as mp
# import joblib
# import google.generativeai as genai
# import requests
# from google import genai
# import os

# # --- CONFIG & LLM SETUP ---
# MODEL_PATH = 'asl_model.pkl'
# THRESHOLD = 8   
# CONF_MIN = 0.65 
# API_KEY = "AIzaSyDxDpuwuVRx81sEl30zwGX8dqNBxPXxZ5g" # Replace with your key

# genai.configure(api_key=API_KEY)
# llm_model = genai.GenerativeModel('gemini-1.5-flash') # Using 1.5 Flash (stable)

# # def fix_grammar(fragments):
# #     if not fragments: return ""
# #     prompt = f"The following are fragments from a sign language user. Turn them into one clear, natural English sentence: {' '.join(fragments)}"
# #     try:
# #         response = llm_model.generate_content(prompt)
# #         return response.text.strip()
# #     except Exception as e:
# #         return f"LLM Error: {e}"

# # --- INITIALIZATION ---
# try:
#     model = joblib.load(MODEL_PATH)
# except:
#     print("Error: Could not find asl_model.pkl.")
#     exit()

# # def fix_grammar(fragments):
# #     if not fragments: return ""
# #     raw_input = ' '.join(fragments)
# #     print(f"--- Sending to LLM: {raw_input} ---") # Check terminal for this!
    
# #     prompt = f"The following are fragments from a sign language user. Turn them into one clear, natural English sentence: {raw_input}"
    
# #     try:
# #         # Adding a request_options timeout so it doesn't hang forever
# #         response = llm_model.generate_content(
# #             prompt, 
# #             request_options={'timeout': 10} 
# #         )
# #         print(f"--- LLM Success: {response.text.strip()} ---")
# #         return response.text.strip()
# #     except Exception as e:
# #         print(f"--- LLM Error: {e} ---")
# #         return f"Error: {e}"

# def fix_grammar(fragments):
#     if not fragments: return ""
#     raw_input = ' '.join(fragments)
#     print(f"--- Attempting HTTP Fallback for: {raw_input} ---")
    
#     # Standard REST API URL
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    
#     headers = {'Content-Type': 'application/json'}
#     payload = {
#         "contents": [{
#             "parts": [{"text": f"Turn these sign language fragments into a natural English sentence: {raw_input}"}]
#         }]
#     }
    
#     try:
#         # 5 second timeout is plenty for a small sentence
#         response = requests.post(url, json=payload, headers=headers, timeout=5)
#         response_data = response.json()
        
#         # Digging into the JSON response
#         result = response_data['candidates'][0]['content']['parts'][0]['text']
#         print(f"--- Success: {result.strip()} ---")
#         return result.strip()
#     except Exception as e:
#         print(f"--- HTTP Error: {e} ---")
#         return "Network Error: Could not reach Gemini"

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)
# sentence_fragments = []
# final_sentence = "" # To display the LLM output
# current_word = ""
# counter = 0

# print("Predictor started. Gesture 'SPACE' to finalize sentence.")

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape
    
#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     row_data = np.zeros(126)
#     prediction_made = False

#     if results.multi_hand_landmarks:
#         for i, hand_lms in enumerate(results.multi_hand_landmarks):
#             if i < 2:
#                 wrist = hand_lms.landmark[0]
#                 hand_coords = []
#                 for lm in hand_lms.landmark:
#                     hand_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
#                 row_data[i*63 : (i+1)*63] = hand_coords
#                 mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
#         probs = model.predict_proba([row_data])
#         conf = np.max(probs)
#         pred = model.predict([row_data])[0]

#         if conf > CONF_MIN:
#             prediction_made = True
#             if pred == current_word:
#                 counter += 1
#             else:
#                 current_word = pred
#                 counter = 0
            
#             # Lock-in word logic
#             if counter == THRESHOLD:
#                 if pred == "space":
#                     # TRIGGER LLM: Process current fragments
#                     if sentence_fragments:
#                         final_sentence = fix_grammar(sentence_fragments)
#                         sentence_fragments = [] # Clear the fragments
#                 elif pred != "none":
#                     if not sentence_fragments or sentence_fragments[-1] != pred:
#                         sentence_fragments.append(pred)
#                         final_sentence = "" # Clear old LLM text when new signs start
#                 counter = 0
#     else:
#         current_word = ""
#         counter = 0

#     # --- UI ---
#     # Top Bar: Shows raw fragments OR the final LLM sentence
#     cv2.rectangle(frame, (0, 0), (w, 70), (40, 40, 40), -1)
    
#     if final_sentence:
#         display_text = f"LLM: {final_sentence}"
#         color = (0, 255, 0) # Green for finished sentence
#     else:
#         display_text = " ".join(sentence_fragments).upper()
#         color = (0, 255, 255) # Yellow for ongoing fragments

#     cv2.putText(frame, display_text, (20, 45), 1, 1.5, color, 2)
    
#     # Bottom Status
#     if prediction_made and current_word != "none":
#         status_color = (0, 0, 255) if current_word == "space" else (0, 255, 0)
#         cv2.putText(frame, f"SIGN: {current_word.upper()}", (20, h-20), 1, 1.5, status_color, 2)

#     cv2.imshow('ASL Two-Hand Predictor', frame)
    
#     key = cv2.waitKey(1)
#     if key == ord('q'): break
#     if key == ord('c'): 
#         sentence_fragments = []
#         final_sentence = ""

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
import joblib
import language_tool_python
import os
import requests  # To talk to your Backend Server

# --- MAC M2 / JAVA OPTIMIZATION ---
os.environ['JAVA_HOME'] = '/opt/homebrew/opt/openjdk'
os.environ['PATH'] = '/opt/homebrew/opt/openjdk/bin:' + os.environ['PATH']

# --- CONFIG ---
MODEL_PATH = 'asl_model.pkl'
THRESHOLD = 8   
CONF_MIN = 0.65 
BACKEND_URL = "http://localhost:5000/speak" # Adjust port if your Node server uses a different one

# --- NLP ENGINE INITIALIZATION ---
print("Initializing Java Grammar Server...")
try:
    tool = language_tool_python.LanguageTool('en-US')
    print("Grammar Engine: ONLINE")
except Exception as e:
    print(f"Grammar Engine failed: {e}. Switching to Simple Mode.")
    tool = None

def send_to_frontend(text):
    """Sends the finalized text to the backend to trigger TTS on the web app."""
    if not text: return
    try:
        # We send a POST request to your Node.js/Express backend
        response = requests.post(BACKEND_URL, json={"message": text}, timeout=1)
        if response.status_code == 200:
            print(f"Successfully broadcasted: {text}")
    except Exception as e:
        print(f"Backend Communication Error: {e}")

def fix_grammar_locally(fragments):
    if not fragments: return ""
    words = [w.lower() for w in fragments]
    
    # 1. Subject Correction
    if words[0] == "me":
        words[0] = "i"
    
    # 2. Verb Injection (State of Being)
    adjectives = ["happy", "sad", "hungry", "thirsty", "tired", "fine"]
    if len(words) >= 2 and words[0] == "i" and words[1] in adjectives:
        words.insert(1, "am")
        
    processed_text = " ".join(words)

    # 3. Phrase Mapping
    replacements = {
        "i want": "I want",
        "where you": "where are you",
        "hello i": "Hello, I am",
    }
    for gloss, correct in replacements.items():
        if gloss in processed_text.lower():
            processed_text = processed_text.lower().replace(gloss, correct)

    # 4. Java Polish
    if tool:
        try:
            processed_text = tool.correct(processed_text)
        except: pass
            
    final = processed_text.strip().capitalize()
    if any(q in final.lower() for q in ["where", "who", "what", "why", "how"]):
        if not final.endswith("?"): final = final.rstrip(".") + "?"
    elif not final.endswith("."):
        final += "."
    return final

# --- MODEL LOADING ---
try:
    model = joblib.load(MODEL_PATH)
    print("ASL Model: LOADED")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sentence_fragments = []
final_sentence = "" 
current_word = ""
counter = 0

print("\n--- ENGINE RUNNING ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    row_data = np.zeros(126) 
    prediction_made = False

    if results.multi_hand_landmarks:
        for i, hand_lms in enumerate(results.multi_hand_landmarks):
            if i < 2:
                wrist = hand_lms.landmark[0]
                coords = []
                for lm in hand_lms.landmark:
                    coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                row_data[i*63 : (i+1)*63] = coords
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
        
        probs = model.predict_proba([row_data])
        conf = np.max(probs)
        pred = model.predict([row_data])[0]

        if conf > CONF_MIN:
            prediction_made = True
            if pred == current_word:
                counter += 1
            else:
                current_word = pred
                counter = 0
            
            if counter == THRESHOLD:
                if pred == "space":
                    if sentence_fragments:
                        final_sentence = fix_grammar_locally(sentence_fragments)
                        # TRIGGER FRONTEND TTS & UI
                        send_to_frontend(final_sentence)
                        sentence_fragments = [] 
                elif pred != "none":
                    if not sentence_fragments or sentence_fragments[-1] != pred:
                        sentence_fragments.append(pred)
                        final_sentence = "" 
                counter = 0
    else:
        current_word = ""
        counter = 0

    # UI Overlays
    cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)
    display_text = final_sentence if final_sentence else " ".join(sentence_fragments).upper()
    color = (0, 255, 0) if final_sentence else (0, 255, 255)
    cv2.putText(frame, display_text, (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

    cv2.imshow('Universal Link - Backend Engine', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    if key == ord('c'): 
        sentence_fragments = []
        final_sentence = ""

cap.release()
cv2.destroyAllWindows()
for i in range(1, 5): cv2.waitKey(1)