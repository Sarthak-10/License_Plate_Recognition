import pandas as pd

model = YOLO('/content/tesseract-5.1.0/runs/detect/train2/weights/best.pt')
model.predict(source='/content/drive/MyDrive/170609_A_Delhi_017.mp4' ,save=True, imgsz=640, conf=0.5,save_crop = True)

