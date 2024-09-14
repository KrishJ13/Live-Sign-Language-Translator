import cv2

"""
Creating Webcam object to enable the use of threading
"""
class Webcam:
    def __init__(self, video_source=0): # Default to webcam
        self.video_source = video_source
        self.capture = cv2.VideoCapture(self.video_source)
        if not self.capture.isOpened():
            raise ValueError("Unable to open video source", self.video_source)

    def get_frame(self):
        if self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # CV2 uses BGR format but Tkinter uses RGB so need to convert
        return None

    def __del__(self):
        if self.capture.isOpened():
            self.capture.release()