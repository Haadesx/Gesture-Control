import cv2
import mediapipe as mp
import pyautogui
import time

# Function to jump in the T-Rex game
def jump():
    pyautogui.press('space')

# Main function
def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a selfie-view display.
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            # Process the image with MediaPipe Hands.
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            # Convert the RGB image back to BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get landmarks
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        landmarks.append((x, y))

                    # Determine gestures
                    # Example: Check if thumb tip is above a certain threshold to jump
                    thumb_tip = landmarks[4]
                    if thumb_tip[1] < 0.2 * image.shape[0]:
                        jump()

            # Show the image.
            cv2.imshow('T-Rex Game', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
