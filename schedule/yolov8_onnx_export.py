# install necessary pecakges
# !pip install ultralytics
# !pip install onnx

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
success = model.export(format="onnx")