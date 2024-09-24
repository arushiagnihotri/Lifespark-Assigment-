import cv2
import time
from hand_tracker import HandTracker


hand_tracker = HandTracker()

def main():
    cap = cv2.VideoCapture(0)
    tap_count = 0
    tapped = False
    start_time = time.time()
    timer_duration = 20  # 20 seconds for the test

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Detect hands in the frame
        hand_landmarks = hand_tracker.detect_hands(frame)
        hand_tracker.draw_hands(frame, hand_landmarks)

        # Check for finger taps
        if hand_tracker.count_finger_taps(hand_landmarks):
            if not tapped:
                tap_count += 1
                tapped = True
        else:
            tapped = False

        # Timer countdown
        elapsed_time = int(time.time() - start_time)
        remaining_time = timer_duration - elapsed_time

        # Display the tap count and timer on the frame
        cv2.putText(frame, f"Taps: {tap_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Time left: {remaining_time}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Finger Tap Test', frame)

        # Exit if the timer runs out
        if remaining_time <= 0:
            break

        # 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Test completed. Total Taps: {tap_count}")

if __name__ == "__main__":
    main()
