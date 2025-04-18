import os
import requests
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
from deepface import DeepFace

app = Flask(__name__)
GEMINI_API_KEY = ""
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Load YOLOv8 model (download yolov8n.pt from Ultralytics site)
yolo_model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def chatbot():
    if 'image' not in request.files or 'query' not in request.form:
        return jsonify({"reply": "Please upload an image and enter your question."})

    image_file = request.files['image']
    user_query = request.form['query']

    # Save uploaded image
    image_path = os.path.join("static", image_file.filename)
    image_file.save(image_path)
    img = Image.open(image_path)

    # YOLO object detection
    results = yolo_model(image_path)
    names = results[0].names
    detected_class_ids = results[0].boxes.cls.int().tolist()
    detected_objects = list({names[obj_id] for obj_id in detected_class_ids})
    object_list = ", ".join(detected_objects) if detected_objects else "nothing"

    # Deepfake detection
    deepfake_result = "Deepfake detection not run."
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            deepfake_result = "This image appears to show a real face."
        else:
            deepfake_result = "This image may be AI-generated or manipulated."
    except Exception as e:
        deepfake_result = f"Deepfake detection error: {str(e)}"

    # Construct prompt for Gemini
    prompt = (
        f"The image contains: {object_list}.\n"
        f"Deepfake Analysis: {deepfake_result}\n"
        f"User asked: '{user_query}'. Respond clearly and concisely."
    )

    # Gemini API payload
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(GEMINI_URL, headers=headers, json=payload)
        result = response.json()
        reply = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No response.")
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"Error contacting Gemini: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)

