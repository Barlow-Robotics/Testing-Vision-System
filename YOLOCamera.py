from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(data="FRC Scorekeeper 2026.v1-trial1.yolo26/data.yaml", epochs=50, imgsz=640, batch=4, device='mps', cache=False, workers=0, patience=15)
