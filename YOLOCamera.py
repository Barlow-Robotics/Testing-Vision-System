from ultralytics import YOLO

model = YOLO("yolo26m.pt")

model.train(data="FRC Scorekeeper 2026.v1-trial1.yolo26/data.yaml", epochs=100, imgsz=640, batch=16, device=0, cache=True, workers=8)
