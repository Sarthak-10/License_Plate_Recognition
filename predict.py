import pandas as pd
from ultralytics import YOLO

model = YOLO('best.pt')
model.predict(source='170609_A_Delhi_017.mp4' ,save=True, imgsz=640, conf=0.5,save_crop = True)

