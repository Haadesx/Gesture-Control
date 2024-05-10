import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to check if the hand is open
def is_hand_open(hand_landmarks):
    # Define finger landmark indices for thumb, index, middle, ring, and pinky fingers
    finger_indices = [mp_hands.HandLandmark.THUMB_TIP,
                      mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP,
                      mp_hands.HandLandmark.PINKY_TIP]
    # Check if all fingers are open
    for index in finger_indices:
        if hand_landmarks.landmark[index].y > hand_landmarks.landmark[index - 1].y:
            return False
    return True

# Function to detect if the hand is in a gesture to accelerate
def is_accelerate_gesture(hand_landmarks):
    # Check if the hand is open
    return is_hand_open(hand_landmarks)

# Function to detect if the hand is in a gesture to turn left
def is_turn_left_gesture(hand_landmarks):
    # Define finger landmark indices for thumb, index, and middle fingers
    finger_indices = [mp_hands.HandLandmark.THUMB_TIP,
                      mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    # Check if the thumb is to the right of the other fingers
    if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x >
        min(hand_landmarks.landmark[index].x for index in finger_indices)):
        return True
    return False

# Function to detect if the hand is in a gesture to turn right
def is_turn_right_gesture(hand_landmarks):
    # Define finger landmark indices for thumb, index, and middle fingers
    finger_indices = [mp_hands.HandLandmark.THUMB_TIP,
                      mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    # Check if the thumb is to the left of the other fingers
    if (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
        max(hand_landmarks.landmark[index].x for index in finger_indices)):
        return True
    return False

# Function to detect if the hand is in a gesture to brake
def is_brake_gesture(hand_landmarks):
    # Define finger landmark indices for index, middle, ring, and pinky fingers
    finger_indices = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      mp_hands.HandLandmark.RING_FINGER_TIP,
                      mp_hands.HandLandmark.PINKY_TIP]
    # Check if all fingers are closed
    for index in finger_indices:
        if hand_landmarks.landmark[index].y < hand_landmarks.landmark[index - 1].y:
            return False
    return True
# Function to control Trackmania based on hand gestures
# Function to control Trackmania based on hand gestures
def control_trackmania(hand_landmarks):
    # Detect current hand gestures
    current_accelerate_state = is_accelerate_gesture(hand_landmarks)
    current_turn_left_state = is_turn_left_gesture(hand_landmarks)
    current_turn_right_state = is_turn_right_gesture(hand_landmarks)
    current_brake_state = is_brake_gesture(hand_landmarks)

    # Send key press events based on hand gestures
    if current_accelerate_state:
        # Up arrow key
        pyautogui.press('up')
    if current_turn_left_state:
        # Left arrow key
        pyautogui.press('left')
    if current_turn_right_state:
        # Right arrow key
        pyautogui.press('right')
    if current_brake_state:
        # Down arrow key
        pyautogui.press('down')
        
        
        
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
                
                # Control Trackmania based on hand gestures
                control_trackmania(hand_landmarks)

        # Display image
        cv2.imshow('Hand Tracking', image)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
