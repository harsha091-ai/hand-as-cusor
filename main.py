import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

current_x, current_y = pyautogui.position()
smoothing_factor = 0.5
pinch_threshold = 30  # Adjust this value to change sensitivity

# Add these variables with the other initializations
previous_middle_y = None
scroll_sensitivity = 2  # Adjust this value to change scroll speed

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            # Existing finger detection for cursor control
            index_finger = landmarks[8]  # Index finger tip
            thumb = landmarks[4]         # Thumb tip
            middle_finger = landmarks[12]  # Middle finger tip
            
            # Get coordinates for all fingers
            index_x = int(index_finger.x * frame_width)
            index_y = int(index_finger.y * frame_height)
            thumb_x = int(thumb.x * frame_width)
            thumb_y = int(thumb.y * frame_height)
            middle_y = int(middle_finger.y * frame_height)

            # Calculate pinch distance (existing click functionality)
            distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

            # Draw circles on fingers
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255))
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 255))
            cv2.circle(frame, (int(middle_finger.x * frame_width), middle_y), 10, (0, 0, 255))

            # Cursor movement (existing code)
            target_x = screen_width/frame_width * index_x
            target_y = screen_height/frame_height * index_y
            current_x = current_x + (target_x - current_x) * smoothing_factor
            current_y = current_y + (target_y - current_y) * smoothing_factor
            pyautogui.moveTo(int(current_x), int(current_y), duration=0.1, _pause=False)

            # Click detection (existing code)
            if distance < pinch_threshold:
                pyautogui.click()
                pyautogui.sleep(0.5)

            # Scroll detection using middle finger
            if previous_middle_y is not None:
                # Check if middle finger is raised (higher than index finger)
                if middle_y < index_y:
                    # Calculate scroll direction and amount
                    scroll_amount = (middle_y - previous_middle_y) * scroll_sensitivity
                    pyautogui.scroll(int(scroll_amount))
            
            previous_middle_y = middle_y

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()