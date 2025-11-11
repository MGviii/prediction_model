from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load("eta_model.joblib")  # Make sure your model is saved with joblib
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": "ETA Predictor v1.0",
        "framework": "Flask"
    }), 200
@app.route("/predict", methods=["POST"])
def predict_eta():
    """
    Expects JSON:
    {
        "busLat": float,
        "busLng": float,
        "speed": float,
        "heading": float,
        "students": [
            { "studentId": "S1", "lastLat": ..., "lastLng": ..., "lastStatus": "check-in" },
            ...
        ]
    }
    Returns:
    {
        "S1": 120.5,
        "S2": 95.2,
        ...
    }
    """
    data = request.get_json()
    bus_lat = data.get("busLat")
    bus_lng = data.get("busLng")
    speed = data.get("speed", 0)
    heading = data.get("heading", 0)
    students = data.get("students", [])

    if not bus_lat or not bus_lng or not students:
        return jsonify({"error": "Missing bus location or students"}), 400

    results = {}
    for student in students:
        student_id = student.get("studentId")
        # Compute distance to next stop: here we use last check-in to determine next stop
        # For simplicity, distance_to_stop = straight-line distance
        stop_lat = student.get("lastLat")
        stop_lng = student.get("lastLng")

        distance_to_stop = ((bus_lat - stop_lat)**2 + (bus_lng - stop_lng)**2)**0.5

        # time_of_day in hours (optional, default 13)
        time_of_day = pd.Timestamp.now().hour

        # Prepare input for model
        features = [[bus_lat, bus_lng, speed, heading, distance_to_stop, time_of_day]]

        eta = model.predict(features)[0]  # predicted in seconds
        results[student_id] = round(float(eta), 2)

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
