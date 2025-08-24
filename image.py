from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict("original.jpg", save=True, conf=0.5)

results[0].show()