import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import time

model = YOLO("best_final.pt")  # Đường dẫn model

cap = None

# Biến toàn cục
label_start_time = {1: None, 2: None}  # Thời điểm bắt đầu đếm label
last_detected_time = 0  # Lưu thời điểm cuối cùng có thông báo
last_label_message = ""  # Lưu thông báo gần nhất

def stop_task():
    global cap, label_start_time, last_detected_time, last_label_message
    if cap is not None:
        cap.release()
        cap = None
    panel.config(image="")
    panel.image = None
    text_label.config(text="")
    label_start_time = {1: None, 2: None}
    last_detected_time = 0
    last_label_message = ""

def detect_objects(frame):
    global label_start_time, last_detected_time, last_label_message

    results = model(frame)
    current_time = time.time()
    message = "Không phát hiện bệnh"
    found_label = None

    for result in results:
        frame = result.plot()
        if result.boxes is not None and result.boxes.cls is not None and len(result.boxes.cls) > 0:
            label = int(result.boxes.cls[0])
            found_label = label

            if label in [1, 2]:
                if label_start_time[label] is None:
                    label_start_time[label] = current_time

                duration = current_time - label_start_time[label]

                if duration <= 30:
                    # Nếu chưa quá 30s thì cập nhật thông báo
                    if label == 1:
                        message = "Người này có triệu chứng ho"
                    elif label == 2:
                        message = "Người này có triệu chứng buồn ngủ"
                    last_detected_time = current_time
                    last_label_message = message
                else:
                    # Nếu đã quá 30s thì reset để chờ nhận diện lại
                    label_start_time[label] = None

                # Reset label còn lại
                other = 2 if label == 1 else 1
                label_start_time[other] = None
        else:
            label_start_time = {1: None, 2: None}

    # Hiển thị thông báo nếu trong 30s gần nhất có phát hiện
    if current_time - last_detected_time <= 30 and last_label_message != "":
        text_label.config(text=last_label_message)
    else:
        text_label.config(text="Không phát hiện bệnh")

    return frame



def load_image():
    stop_task()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    frame = detect_objects(img)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)

    panel.config(image=img)
    panel.image = img

def load_video():
    stop_task()
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)

    def update_video():
        if cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            stop_task()
            return

        frame = detect_objects(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        root.after(30, update_video)

    update_video()

def load_camera():
    stop_task()
    global cap
    cap = cv2.VideoCapture(0)

    def update_camera():
        if cap is None:
            return

        ret, frame = cap.read()
        if not ret:
            stop_task()
            return

        frame = detect_objects(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img

        root.after(10, update_camera)

    update_camera()

root = tk.Tk()
root.title("Object Detection GUI")

btn_image = tk.Button(root, text="Nhận diện qua hình ảnh", command=load_image)
btn_image.pack(pady=5)

btn_video = tk.Button(root, text="Nhận diện qua video", command=load_video)
btn_video.pack(pady=5)

btn_camera = tk.Button(root, text="Nhận diện qua camera", command=load_camera)
btn_camera.pack(pady=5)

btn_stop = tk.Button(root, text="Dừng tác vụ", command=stop_task)
btn_stop.pack(pady=5)

panel = tk.Label(root)
panel.pack()

text_label = tk.Label(root, text="", font=("Arial", 14))
text_label.pack()

root.mainloop()
