from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict
    results = model(frame)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Show live window
    cv2.imshow("Brain Tumor Detection", annotated_frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
