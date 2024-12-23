import cv2                  # OpenCV library for computer vision tasks.
import numpy as np          # Library for numerical operations.
import os                   # Operating system library for file handling.
import tkinter as tk        # Python's standard GUI library for creating the user interface.
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk # python Imaging Library for handling and processing images.


class DetectionApp:
    def __init__(self, root):   # Initializes the main application window with a title, size, and background color.
        self.root = root
        self.root.title("Face & Object Detection App")
        self.root.geometry("900x700")
        self.root.configure(bg="#282c34")

        # Video display area
        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Buttons
        self.open_face_detection_btn = tk.Button(root, text="Face Detection", command=self.open_face_detection,
                                                 bg="#61afef", fg="white", font=("Arial", 12, "bold"))
        self.open_face_detection_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.open_object_detection_btn = tk.Button(root, text="Object Detection", command=self.open_object_detection,
                                                   bg="#61afef", fg="white", font=("Arial", 12, "bold"))
        self.open_object_detection_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.select_file_btn = tk.Button(root, text="Select File (Photo/Video)", command=self.select_file,
                                         bg="#61afef", fg="white", font=("Arial", 12, "bold"))
        self.select_file_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.start_detection_btn = tk.Button(root, text="Start Detection", command=self.start_file_detection,
                                             bg="#98c379", fg="white", font=("Arial", 12, "bold"))
        self.start_detection_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.close_camera_btn = tk.Button(root, text="Close Camera", command=self.close_camera,
                                          bg="#e06c75", fg="white", font=("Arial", 12, "bold"))
        self.close_camera_btn.pack(side=tk.LEFT, padx=10, pady=10)

        self.quit_btn = tk.Button(root, text="Quit", command=self.quit_app,
                                  bg="#e06c75", fg="white", font=("Arial", 12, "bold"))
        self.quit_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # Variables These variables are used to manage the state of the application 
        # (e.g., camera capture, running status, file path, etc.).
        self.cap = None
        self.running = False
        self.file_path = None
        self.is_video = False
        self.output_folder = "output"
        self.detection_mode = None  # 'face' or 'object'

        # Load YOLO model for object detection
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            messagebox.showerror("Error", f"Error loading YOLO model: {e}")
            self.quit_app()

        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def open_face_detection(self):
        """Starts face detection using webcam."""
        self.detection_mode = 'face'
        self.start_camera()

    def open_object_detection(self):
        """Starts object detection using webcam."""
        self.detection_mode = 'object'
        self.start_camera()

    def start_camera(self):
        """Opens the camera."""
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open the webcam.")
                return

            self.running = True
            self.update_camera_feed()

    def update_camera_feed(self):
        """Updates the camera feed based on the detection mode."""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.detection_mode == 'face':
                    frame = self.detect_faces(frame)
                elif self.detection_mode == 'object':
                    frame = self.detect_objects_in_frame(frame)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                self.video_label.after(10, self.update_camera_feed)

    def detect_faces(self, frame):
        """Detects faces in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame

    def detect_objects_in_frame(self, frame):
        """Performs object detection on a frame."""
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def select_file(self):
        """Opens a file dialog to select a photo or video."""
        file_path = filedialog.askopenfilename(title="Select File", filetypes=[("Photo/Video Files", "*.mp4 *.avi *.jpg *.jpeg *.png")])
        if file_path:
            self.file_path = file_path
            self.is_video = file_path.endswith(('.mp4', '.avi'))
            messagebox.showinfo("File Selected", f"File selected: {os.path.basename(file_path)}")

    def start_file_detection(self):
        """Starts detection on the selected file."""
        if not self.file_path:
            messagebox.showerror("Error", "No file selected.")
            return
        if self.is_video:
            self.detect_objects_in_video()
        else:
            self.detect_objects_in_photo()

    def detect_objects_in_photo(self):
        """Detects objects in a photo."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        frame = cv2.imread(self.file_path)
        frame = self.detect_objects_in_frame(frame)

        output_path = os.path.join(self.output_folder, f"detected_{os.path.basename(self.file_path)}")
        cv2.imwrite(output_path, frame)
        messagebox.showinfo("Success", f"Detection completed! Output saved at: {output_path}")

    def detect_objects_in_video(self):
        """Detects objects in a video."""
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        cap = cv2.VideoCapture(self.file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_folder, f"detected_{os.path.basename(self.file_path)}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_objects_in_frame(frame)
            out.write(frame)

        cap.release()
        out.release()
        messagebox.showinfo("Success", f"Detection completed! Output saved at: {output_path}")

    def close_camera(self):
        """Closes the camera."""
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image="")
        self.detection_mode = None

    def quit_app(self):
        """Quits the application."""
        self.close_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
