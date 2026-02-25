 
import cv2
from ultralytics import YOLO
import math
import mss
import numpy as np

CONF_THRES = 0.7
real_diamter_of_ball = 5.91 * 0.0254  # Convert inches to meters
Horintoal_FOV = 63.5

def compile_image(frame):
    results = model(frame, stream=True)
    height, image_width, channels = frame.shape

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   
            conf = float(box.conf[0])                
            cls = int(box.cls[0])                       
            label = f"{model.names[cls]} {conf:.2f}" 

            if conf > CONF_THRES: 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)

                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                bbox_width = x2 - x1
                f = image_width / (2 * (math.tan(math.radians(Horintoal_FOV / 2))))
                distance = (real_diamter_of_ball * f) / bbox_width
                
                distance *= 39.3701
                distance_text = f"{distance:.2f} in"
                cv2.putText(frame, distance_text, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                
    cv2.imshow("YOLOv8 Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()
        
        
        
model = YOLO('runs/detect/train20/yolov8npretrainedballmodel.pt')
mode = 'camera'
if mode == 'camera':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam didn't open :(")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        compile_image(frame)
        
    


    cap.release()
    cv2.destroyAllWindows()
elif mode == 'screen':
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            screenshot = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            compile_image(frame)

