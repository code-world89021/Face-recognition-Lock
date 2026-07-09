import cv2
import face_recognition
import numpy as np
import pyttsx3
import os
import threading

# =========================
# TEXT TO SPEECH
# =========================

engine = pyttsx3.init()

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()

# =========================
# LOAD KNOWN FACE
# =========================

image_path = r"My_face.jpg"

if not os.path.exists(image_path):
    print("ERROR: Image not found!")
    exit()

known_image = face_recognition.load_image_file(image_path)

encodings = face_recognition.face_encodings(known_image)

if len(encodings) == 0:
    print("ERROR: No face found in image!")
    exit()

known_encoding = encodings[0]

known_faces = [known_encoding]
known_names = ["Prince"]

# =========================
# CAMERA DETECTION
# =========================

video = None

for i in range(5):

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    if cap.isOpened():
        video = cap
        print(f"Camera found at index {i}")
        break

    cap.release()

if video is None:
    print("ERROR: No camera found")
    exit()

# =========================
# VARIABLES
# =========================

frame_count = 0

face_status = "SCANNING..."

last_voice_status = ""

print("Face Lock Started")

# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = video.read()

    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1

    # -------------------------
    # Scan every 5th frame only
    # -------------------------
    if frame_count % 5 == 0:

        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=0.25,
            fy=0.25
        )

        rgb_small_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        try:

            face_encodings = face_recognition.face_encodings(
                rgb_small_frame
            )

            if len(face_encodings) == 0:

                face_status = "NO FACE"

            else:

                match_found = False

                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(
                        known_faces,
                        face_encoding,
                        tolerance=0.45
                    )

                    face_distance = face_recognition.face_distance(
                        known_faces,
                        face_encoding
                    )

                    best_match = np.argmin(face_distance)

                    if matches[best_match]:

                        face_status = "ACCESS GRANTED"
                        match_found = True

                        if last_voice_status != "GRANTED":
                            print("ACCESS GRANTED")
                            speak("Welcome Prince")
                            last_voice_status = "GRANTED"

                        break

                if not match_found:

                    face_status = "ACCESS DENIED"

                    if last_voice_status != "DENIED":
                        print("ACCESS DENIED")
                        speak("Unknown face. How are you mister?")
                        last_voice_status = "DENIED"

        except Exception as e:
            print("Face Recognition Error:", e)

    # =========================
    # DISPLAY STATUS
    # =========================

    if face_status == "ACCESS GRANTED":
        color = (0, 255, 0)

    elif face_status == "ACCESS DENIED":
        color = (0, 0, 255)

    else:
        color = (0, 255, 255)

    cv2.putText(
        frame,
        face_status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Face Lock System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEAN EXIT
# =========================

video.release()
cv2.destroyAllWindows()
