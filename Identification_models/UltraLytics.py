import ultralytics
from ultralytics import YOLO

# Run system checks
ultralytics.checks()

# Load the YOLO model 
model = YOLO('yolov8n.pt')# here we can use any previous version to check the scope and the measure the development in these models

# Perform prediction on the given source image
results = model.predict(source='https://ultralytics.com/images/zidane.jpg')

# Display results
print(results)
