from roboflow import Roboflow
from ultralytics import YOLO
import torch

__version__ = '0.2.1'

rf = Roboflow(api_key="f1sOJWjo8ZDrK6zgiZJ1")
# project = rf.workspace("blood-cells").project("traffic-pj8qd")
# dataset = project.version(1).download("yolov8")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = YOLO('yolov8n.pt').to(device)

model.train(data='data.yaml', epochs=100, imgsz=640)