from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict("test_image.jpg", save=True, conf=0.5)

results[0].show() 
