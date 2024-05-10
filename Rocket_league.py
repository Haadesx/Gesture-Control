import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to count opened fingers for both hands
def count_opened_fingers(hand_landmarks):
    # Define finger landmark indices
    finger_tip_indices = [4, 8, 12, 16, 20]
    # Count opened fingers
    opened_fingers = 0
    for finger_tip_index in finger_tip_indices:
        # Check if the finger is opened (y-coordinate of finger tip landmark is higher than y-coordinate of its adjacent landmark)
        if hand_landmarks.landmark[finger_tip_index].y < hand_landmarks.landmark[finger_tip_index - 1].y:
            opened_fingers += 1
    # Return 0 if no fingers are opened
    return opened_fingers

# Function to control Rocket League based on finger counts
def control_rocket_league(finger_counts):
    if finger_counts == 0:
        # Accelerate forward
        pyautogui.keyDown('w')
    elif finger_counts == 1:
        # Turn right
        pyautogui.keyDown('right')
    elif finger_counts == 2:
        # Turn left
        pyautogui.keyDown('left')
    elif finger_counts == 3:
        # Activate boost (shift)
        pyautogui.keyDown('shift')
    elif finger_counts == 4:
        # Perform a right-click action (e.g., jump)
        pyautogui.rightClick()
    elif finger_counts == 5:
        # Perform a double right-click action (e.g., special move)
        pyautogui.rightClick()
        pyautogui.rightClick()
    else:
        # No action
        pass

# Initialize MediaPipe Hands
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip image horizontally
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count opened fingers for each hand
                opened_fingers = count_opened_fingers(hand_landmarks)
                
                # Control Rocket League based on finger counts
                control_rocket_league(opened_fingers)

                # Display count on screen for each hand
                cv2.putText(image, str(opened_fingers),
                            (int(hand_landmarks.landmark[0].x * image.shape[1]),
                             int(hand_landmarks.landmark[0].y * image.shape[0]) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 0), 2, cv2.LINE_AA)
        
        # Display image
        cv2.imshow('Hand Tracking', image)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
