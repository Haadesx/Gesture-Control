# import cv2
# import mediapipe as mp
# import pyautogui

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Function to check if the hand is open
# def is_hand_open(hand_landmarks):
#     # Define finger landmark indices for thumb, index, middle, ring, and pinky fingers
#     finger_indices = [mp_hands.HandLandmark.THUMB_TIP,
#                       mp_hands.HandLandmark.INDEX_FINGER_TIP,
#                       mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
#                       mp_hands.HandLandmark.RING_FINGER_TIP,
#                       mp_hands.HandLandmark.PINKY_TIP]
#     # Check if all fingers are open
#     for index in finger_indices:
#         if hand_landmarks.landmark[index].y > hand_landmarks.landmark[index - 1].y:
#             return False
#     return True

# # Function to control Trackmania based on hand positions
# def control_trackmania(left_hand_landmarks, right_hand_landmarks):
#     # Initialize variables to track hand positions
#     left_hand_open = False
#     right_hand_open = False

#     # Determine if left hand is open
#     if left_hand_landmarks:
#         left_hand_open = is_hand_open(left_hand_landmarks)

#     # Determine if right hand is open
#     if right_hand_landmarks:
#         right_hand_open = is_hand_open(right_hand_landmarks)

#     # Control Trackmania based on hand positions
#     if left_hand_open and right_hand_open:
#         # Press 'W' to move forward
#         pyautogui.press('w')
#     elif left_hand_open and not right_hand_open:
#         # Turn left by pressing 'A'
#         pyautogui.press('a')
#     elif not left_hand_open and right_hand_open:
#         # Turn right by pressing 'D'
#         pyautogui.press('d')
#     else:
#         # Press 'S' to move backward
#         pyautogui.press('s')

# # Initialize MediaPipe Hands
# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert BGR to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Flip image horizontally
#         image = cv2.flip(image, 1)
        
#         # Set flag
#         image.flags.writeable = False
        
#         # Detections
#         results = hands.process(image)
        
#         # Set flag to true
#         image.flags.writeable = True

#         # RGB 2 BGR
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         if results.multi_hand_landmarks:
#             # Initialize variables to store left and right hand landmarks
#             left_hand_landmarks = None
#             right_hand_landmarks = None

#             # Loop through detected hand landmarks
#             for hand_landmarks in results.multi_hand_landmarks:
#                 # Check if hand is left or right based on the x-coordinate of the wrist
#                 if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
#                     left_hand_landmarks = hand_landmarks
#                 else:
#                     right_hand_landmarks = hand_landmarks

#             # Control Trackmania based on hand positions
#             control_trackmania(left_hand_landmarks, right_hand_landmarks)

#         # Display image
#         cv2.imshow('Hand Tracking', image)

#         # Break loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()




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

# Function to control Trackmania based on hand positions
def control_trackmania(left_hand_landmarks, right_hand_landmarks):
    # Initialize variables to track hand positions
    left_hand_open = False
    right_hand_open = False

    # Determine if left hand is open
    if left_hand_landmarks:
        left_hand_open = is_hand_open(left_hand_landmarks)

    # Determine if right hand is open
    if right_hand_landmarks:
        right_hand_open = is_hand_open(right_hand_landmarks)

    # Continuous key pressing loop
    while True:
        # Press 'W' to move forward
        if left_hand_open and right_hand_open:
            pyautogui.press('w')
        # Turn left by pressing 'A'
        elif left_hand_open and not right_hand_open:
            pyautogui.press('a')
        # Turn right by pressing 'D'
        elif not left_hand_open and right_hand_open:
            pyautogui.press('d')
        # Press 'S' to move backward
        else:
            pyautogui.press('s')

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
            # Initialize variables to store left and right hand landmarks
            left_hand_landmarks = None
            right_hand_landmarks = None

            # Loop through detected hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if hand is left or right based on the x-coordinate of the wrist
                if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:
                    left_hand_landmarks = hand_landmarks
                else:
                    right_hand_landmarks = hand_landmarks

            # Control Trackmania based on hand positions
            control_trackmania(left_hand_landmarks, right_hand_landmarks)

        # Display image
        cv2.imshow('Hand Tracking', image)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
