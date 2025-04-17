from ultralytics import YOLO

try:
    # Try loading the model with the absolute path
    yolo_model = YOLO("C:\\Users\\krbik\\OneDrive\\Desktop\\image_chatbot\\yolov5s-cls.pt")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
