import tkinter as tk
from PIL import Image, ImageTk
from webcam import Webcam
import threading

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Initialise webcam
        self.webcam = Webcam()

        # Create tkinter canvas for video feed
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Creates button to close the application
        self.exit_button = tk.Button(window, text="Quit", command=self.window.quit)
        self.exit_button.pack()

        self.delay = 15
        self.update_frame()

    def update_frame(self):
        # Fetch frame from webcam
        frame = self.webcam.get_frame()
        if frame is not None:
            # Convert frame to ImageTk format
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # Repeat every self.delay
        self.window.after(self.delay, self.update_frame)

def start_gui():
    root = tk.Tk()
    app = WebcamApp(root, "Sign Language Translator")
    root.mainloop()

if __name__ == "__main__":
    # Run GUI in main thread
    gui_thread = threading.Thread(target=start_gui)
    gui_thread.start()
