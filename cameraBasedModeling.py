 
import cv2
from ultralytics import YOLO

CONF_THRES = 0.78

model = YOLO('yolov8npretrainedballmodel.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam didn't open :(")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

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

    cv2.imshow("YOLOv8 Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()