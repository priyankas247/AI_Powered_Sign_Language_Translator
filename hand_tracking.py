import cv2
import mediapipe as mp
import pyttsx3

# Initialize Mediapipe and TTS
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Sign Language Mapping
SIGN_LANGUAGE_MAP = {
    (0, 0, 0, 0, 0): "Fist (A)",
    (1, 1, 1, 1, 1): "Open Palm (B)",
    (0, 1, 1, 0, 0): "V Sign (Victory)",
    (1, 0, 0, 0, 1): "Call Me 🤙",
}

def get_finger_positions(hand_landmarks):
    """ Returns a list indicating which fingers are up (1) or down (0). """
    fingers = []
    
    # Thumb
    fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    
    # Other fingers
    for tip in [8, 12, 16, 20]:  # Index, Middle, Ring, Pinky
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    return fingers

def recognize_sign(fingers):
    """ Matches finger positions to a predefined sign. """
    return SIGN_LANGUAGE_MAP.get(tuple(fingers), "Unknown Sign")

# Open Camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    sentence = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame!")
            break

        print("Frame Captured!")  # Debugging

        # Convert BGR to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Check if a hand is detected
        if results.multi_hand_landmarks:
            print("Hand Detected!")  # Debugging
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = get_finger_positions(hand_landmarks)

                # Recognize Sign
                sign = recognize_sign(fingers)
                print("Recognized Sign:", sign)  # Debugging

                if sign != "Unknown Sign":
                    sentence += sign + " "

                    # Speak the recognized sign
                    engine.say(sign)
                    engine.runAndWait()

                # Display sign text
                cv2.putText(frame, sign, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display sentence history
        cv2.putText(frame, sentence, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Hand Tracking - Sign Language", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
