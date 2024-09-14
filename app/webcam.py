import cv2  # OpenCV for webcam access and image processing
import mediapipe as mp  # MediaPipe for hand landmark detection
import preprocess  # Custom module for preprocessing hand images

# Webcam class handles capturing video feed, detecting hand landmarks, and preprocessing images
class Webcam:
    def __init__(self, video_source=0):  # Default to using the system's webcam (video_source=0)
        self.video_source = video_source
        self.capture = cv2.VideoCapture(self.video_source)  # Open the video source (webcam)
        self.show_landmarks = False  # Flag to toggle the display of hand landmarks

        # Check if the video source was successfully opened
        if not self.capture.isOpened():
            raise ValueError("Unable to open video source", self.video_source)

        # Initialize MediaPipe Hands for hand landmark detection
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,  # Real-time detection (not for static images)
            max_num_hands=1,  # Detect a maximum of one hand
            min_detection_confidence=0.5  # Confidence threshold for detection
        )

        # MediaPipe utility for drawing hand landmarks and connections
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    # Method to capture and process each frame from the webcam
    def get_frame(self):
        # Check if the webcam capture is open
        if self.capture.isOpened():
            ret, frame = self.capture.read()  # Read a frame from the webcam
            if not ret or frame is None:  # Check if the frame is valid
                return None

            # Get frame dimensions (height, width, and channels)
            height, width, channels = frame.shape
            frame_no_landmarks = frame.copy()  # Create a copy of the frame without landmarks

            # Convert the frame to RGB and process it using MediaPipe to detect hands
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # If hand landmarks are detected, process them
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Initialize bounding box values for hand region
                    x_max, y_max = 0, 0
                    x_min, y_min = width, height

                    # Calculate the bounding box around the detected hand landmarks
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * width), int(landmark.y * height)

                        # Update min and max coordinates if they are within valid frame dimensions
                        if 0 <= x < width and 0 <= y < height:
                            x_max = max(x, x_max)
                            x_min = min(x, x_min)
                            y_max = max(y, y_max)
                            y_min = min(y, y_min)

                    # Adjust the bounding box to make it larger, ensuring it stays within frame boundaries
                    x_min = max(0, x_min - 40)
                    y_min = max(0, y_min - 40)
                    x_max = min(width, x_max + 40)
                    y_max = min(height, y_max + 40)

                    # Draw the bounding box around the detected hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Copy the current frame for further processing
                    frame_to_process = frame.copy()

                    # Preprocess the hand image by cropping it using the bounding box and fixing its size
                    frame_to_process = preprocess.hand_image(x_min + 20, y_min + 20, x_max - 20, y_max - 20, frame_to_process)
                    frame_to_process = preprocess.fix_size(frame_to_process)

                    # Display the processed hand image in a separate window
                    cv2.imshow("Hand Landmarks", frame_to_process)

            # If landmark display is enabled, draw landmarks on the original frame
            if self.show_landmarks and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks and connections between them
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        # Draw the landmarks in green with specific thickness
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        # Draw the hand connections in red
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                # Return the frame with hand landmarks drawn, converted to RGB for compatibility
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Return the frame without landmarks, converted to RGB
                return cv2.cvtColor(frame_no_landmarks, cv2.COLOR_BGR2RGB)
        return None  # Return None if the webcam is not open

    # Toggle method to switch between showing and hiding hand landmarks
    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks

    # Destructor to release the webcam resource when the object is destroyed
    def __del__(self):
        # If the webcam is still open, release it and close any OpenCV windows
        if self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()
