## git ultralytics
!git clone https://github.com/ultralytics/ultralytics.git

## install ultralytics and roboflow dependancies
%pip install ultralytics==8.0.196
%pip install roboflow

## These line of code is produced by Roboflow
## Your model can be deployed via Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="your-API-key")
project = rf.workspace("roboflow-xuntf").project("YOUR-PROJECT")
version = project.version(1)
dataset = version.download("yolov8")

from ultralytics import YOLO
import ultralytics
## check the version of ultralytics, pytorch, cuda, and CPU/GPU model
ultralytics.checks()
## building a model from scratch
model = YOLO("yolov8n.yaml")

## change the path in data.yaml if needed
data_path = "/ultralytics/Coca-cola-Detection-1/data.yaml"

## batch can be set to 64 and epochs can be set greater
## device='cpu', device='mps' for M1/M2, device=0,1,2,or3 for GPU
model.train(data=data_path, batch=32,epochs=25,device=2, val=False)

## get the path to the trained model
## and check the trained model performance with the validation set
model = YOLO("PATH/TO/train/weights/best.pt")
model.val(data=data_path)

## tuning the trained model
## parameters can be modified
model.tune(data=data_path, epochs=20, iterations=50, optimizer='AdamW', plots=False, save=False, val=False)

## get the path to the tuned model
## and check the trained model performance with the validation set
model = YOLO("PATH/TO/tune/weights/best.pt")
model.val(data=data_path)

## get the path to the test images
## and predict on the testing set
test_set="PATH/TO/test/images"
## thresholds can be changed confidence level,
model.predict(data=data_path, conf=0.5, source=test_set, save=True)

## deploy your model to Roboflow
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"PATH/TO/runs/detect/train OR tune/")

## TRY YOUR MODEL!!
webcam = model(source=0,save=False, show=True)
