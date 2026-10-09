import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

print("Brooch AI Started! Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], verbose=False)
    person_count = len(results[0].boxes)
    
    if person_count > 3:
        status_text = f"Crowded! {person_count} persons"
        color = (0, 0, 255)
    else:
        status_text = "Path Clear"
        color = (0, 255, 0)

    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, status_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Digital Eyes - Smart Brooch AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()