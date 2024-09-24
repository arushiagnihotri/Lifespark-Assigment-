import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        hand_landmarks = result.multi_hand_landmarks
        return hand_landmarks

    def draw_hands(self, frame, hand_landmarks):
        if hand_landmarks:
            for landmarks in hand_landmarks:
                self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

    def count_finger_taps(self, landmarks, threshold=0.05):
        if landmarks:
            landmarks = landmarks[0].landmark
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # Calculate distance between thumb and index finger tips
            distance = np.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2 +
                (thumb_tip.z - index_tip.z) ** 2
            )
            
            # Check if the distance is smaller than the threshold (indicating a tap)
            return distance < threshold
        return False
