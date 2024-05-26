from ultralytics import YOLO
model = YOLO('yolov8m-seg.pt')  


results = model.train(data='data_custom.yaml', epochs=25, imgsz=640)