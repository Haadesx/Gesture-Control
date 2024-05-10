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

# Function to control Trackmania based on hand gestures
def control_trackmania(left_hand_open, right_hand_open):
    if left_hand_open and right_hand_open:
        # Both hands open: Accelerate
        pyautogui.keyDown('w')
    elif left_hand_open and not right_hand_open:
        # Left hand open, right hand closed: Turn left
        pyautogui.keyDown('a')
    elif right_hand_open and not left_hand_open:
        # Right hand open, left hand closed: Turn right
        pyautogui.keyDown('d')
    else:
        # Both hands closed: Brake
        pyautogui.keyDown('s')

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
        
        left_hand_open = False
        right_hand_open = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Count opened fingers for each hand
                opened_fingers = count_opened_fingers(hand_landmarks)
                
                # Determine which hand is open
                if hand_landmarks == results.multi_hand_landmarks[0]:
                    left_hand_open = True if opened_fingers > 0 else False
                else:
                    right_hand_open = True if opened_fingers > 0 else False

        # Control Trackmania based on hand gestures
        control_trackmania(left_hand_open, right_hand_open)

        # Display image
        cv2.imshow('Hand Tracking', image)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
