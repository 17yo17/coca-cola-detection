from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")

data_path = "/ultralytics/Coca-cola-Detection-1/data.yaml"

model.train(data=data_path, epochs=3)

model_path="/ultralytics/runs/detect/train/weights/best.pt"

model.val(model=model_path, data=data_path, epochs=3)

test_dataset_path="/ultralytics/Coca-cola-Detection-1/test/images"

model.predict(model=model_path, data=data_path, conf=0.5, source=test_dataset_path, save=True)

webcam = model(source=0,save=False, show=True)
