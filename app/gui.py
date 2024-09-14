import tkinter as tk
from PIL import Image, ImageTk
from webcam import Webcam


# Class representing the webcam application with a GUI interface
class WebcamApp:
    def __init__(self, window, window_title):
        # Initialize the main window and set its title
        self.window = window
        self.window.title(window_title)

        # Initialize the webcam object (assumes Webcam class handles webcam input)
        self.webcam = Webcam()

        # Create a Tkinter canvas where the webcam feed will be displayed
        # The canvas size is set to 640x480 pixels
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()  # Pack the canvas widget into the window

        # Create a button to close the application
        self.exit_button = tk.Button(window, text="Quit", command=self.window.quit)
        self.exit_button.pack()  # Pack the button into the window

        # Create a button to toggle the viewing of hand landmarks (or other preprocessed images)
        # This is assumed to be a feature handled by the Webcam class
        self.landmark_button = tk.Button(window, text="Toggle Hand Landmarks", command=self.webcam.toggle_landmarks)
        self.landmark_button.pack()  # Pack the button into the window

        # Set a delay (in milliseconds) for how often the webcam feed updates
        # This determines the refresh rate of the video feed
        self.delay = 15  # ~66 FPS (1000ms / 15ms = ~66 frames per second)

        # Start the frame update loop
        self.update_frame()

    # Method to update the webcam frame and display it on the canvas
    def update_frame(self):
        # Get the latest frame from the webcam
        frame = self.webcam.get_frame()

        if frame is not None:
            # Convert the webcam frame (assumed to be in NumPy array format) to an ImageTk format
            # ImageTk is necessary for displaying the image in Tkinter
            image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=image)

            # Clear the previous image on the canvas (if any) and display the new one
            self.canvas.delete('all')  # Remove the previous image from the canvas
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)  # Place the new image at the top-left corner

        # Schedule the next frame update after 'self.delay' milliseconds
        self.window.after(self.delay, self.update_frame)


# Function to start the GUI application
def start_gui():
    # Create the main window (root) for the Tkinter application
    root = tk.Tk()

    # Create an instance of WebcamApp, passing the root window and the window title
    app = WebcamApp(root, "Sign Language Translator")

    # Start the Tkinter main event loop (keeps the application running)
    root.mainloop()


# If this script is run directly, start the GUI application
if __name__ == "__main__":
    start_gui()
