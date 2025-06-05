import cv2
import numpy as np
import os
import time
from gtts import gTTS
import tempfile
import pygame

pygame.mixer.init()

SPECIAL_GREETINGS = {
    "Abhishek": "Welcome Abhishek Sir, How are you?"  #Change with your Dataset name
}

def verify_cascade():
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(path):
        print("ERROR: Haar cascade not found!")
        return None
    return cv2.CascadeClassifier(path)

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def speak_greeting(text):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_filename = temp_file.name
        temp_file.close()

        tts = gTTS(text=text, lang='en')
        tts.save(temp_filename)

        pygame.mixer.music.load(temp_filename)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        try:
            os.unlink(temp_filename)
        except:
            pass

    except Exception as e:
        print(f"Error playing greeting: {e}")

def recognize_faces():
    face_detector = verify_cascade()
    if face_detector is None:
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('face_model.yml')
        id_to_name = np.load('labels.npy', allow_pickle=True).item()
        print("Model and labels loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    last_greeting = {}
    greeting_cooldown = 30  
    face_in_dataset = 0
    confidence_threshold = 110
    unknown_threshold = 150
    consecutive_matches = 0
    required_matches = 2

    print("Starting face recognition...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Couldn't capture frame")
            break

        gray = preprocess_image(frame)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(70, 70),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        current_time = time.time()
        face_in_dataset = 0

        for (x, y, w, h) in faces:
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_roi = cv2.resize(gray[y1:y2, x1:x2], (200, 200))
            id, confidence = recognizer.predict(face_roi)

            if confidence < confidence_threshold and id in id_to_name:
                name = id_to_name[id]
                consecutive_matches += 1
                
                if consecutive_matches >= required_matches:
                    color = (0, 255, 0)
                    face_in_dataset = 1

                    if name in SPECIAL_GREETINGS:
                        if name not in last_greeting or (current_time - last_greeting[name]) > greeting_cooldown:
                            greeting_text = SPECIAL_GREETINGS[name]
                            print(f"Greeting recognized person: {name}")
                            speak_greeting(greeting_text)
                            last_greeting[name] = current_time
                else:
                    name = "Processing..."
                    color = (0, 255, 255)
                    face_in_dataset = 0
            elif confidence < unknown_threshold:
                name = "Unknown"
                color = (0, 0, 255)
                face_in_dataset = 0
                consecutive_matches = 0
            else:
                name = "Unknown"
                color = (0, 0, 255)
                face_in_dataset = 0
                consecutive_matches = 0

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{name} {confidence:.1f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.putText(frame, f"Face in dataset: {face_in_dataset}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

if __name__ == "__main__":
    recognize_faces()
