"""
Python ML Microservice for ETA Predictions
Deploy this on Render, Railway, or Heroku
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime, timedelta
from geopy.distance import geodesic
import os

app = Flask(__name__)
CORS(app)  # Allow Node.js backend to call this

class ETAPredictorService:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.student_patterns = {}
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.getenv('MODEL_PATH', 'realtime_eta_model.joblib')
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.student_patterns = model_data.get('patterns', {})
                self.model_loaded = True
                print(f"✓ Model loaded successfully with {len(self.models)} student models")
            else:
                print("⚠️  Model file not found, using fallback predictions")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points"""
        return geodesic((lat1, lon1), (lat2, lon2)).km
    
    def calculate_speed_from_history(self, gps_history):
        """Calculate current speed from GPS history"""
        if not gps_history or len(gps_history) < 2:
            return 20
        
        # Get recent points (last 5 minutes)
        current_time = datetime.now().timestamp() * 1000
        cutoff_time = current_time - (5 * 60 * 1000)
        
        recent_points = []
        for timestamp, data in gps_history.items():
            if data.get('timestamp', 0) >= cutoff_time:
                recent_points.append(data)
        
        if len(recent_points) < 2:
            return 20
        
        # Sort by timestamp
        recent_points.sort(key=lambda x: x.get('timestamp', 0))
        
        speeds = []
        for i in range(1, len(recent_points)):
            prev = recent_points[i-1]
            curr = recent_points[i]
            
            distance = self.haversine_distance(
                prev.get('Latitude', 0),
                prev.get('Longitude', 0),
                curr.get('Latitude', 0),
                curr.get('Longitude', 0)
            )
            
            time_diff = (curr.get('timestamp', 0) - prev.get('timestamp', 0)) / 1000 / 3600
            
            if time_diff > 0 and distance > 0:
                speed = distance / time_diff
                if 0 < speed < 100:
                    speeds.append(speed)
        
        return np.mean(speeds) if speeds else 20
    
    def detect_traffic(self, current_speed, historical_speed):
        """Detect traffic congestion"""
        if historical_speed == 0:
            return 1.0
        
        ratio = current_speed / historical_speed
        
        if ratio < 0.3:
            return 2.5
        elif ratio < 0.5:
            return 2.0
        elif ratio < 0.7:
            return 1.5
        else:
            return 1.0
    
    def predict(self, student_id, bus_location, student_destination, 
                gps_history, emergency_active=False):
        """Main prediction function"""
        
        # Calculate distance
        distance = self.haversine_distance(
            bus_location.get('lat', 0),
            bus_location.get('lng', 0),
            student_destination.get('lat', 0),
            student_destination.get('lng', 0)
        )
        
        # Calculate current speed
        current_speed = self.calculate_speed_from_history(gps_history)
        
        # Get time features
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        day_of_week = now.weekday()
        is_peak_hour = 1 if (6 <= hour <= 9) or (15 <= hour <= 18) else 0
        
        # If we have a trained model for this student
        if self.model_loaded and student_id in self.models:
            # Get historical speed
            pattern = self.student_patterns.get(student_id, {})
            historical_speed = pattern.get('avg_speed', 25)
            
            # Traffic detection
            traffic_factor = self.detect_traffic(current_speed, historical_speed)
            
            # Prepare features
            features = np.array([[
                distance,
                current_speed,
                traffic_factor,
                hour,
                minute,
                day_of_week,
                is_peak_hour,
                0  # estimated_stops
            ]])
            
            # Scale and predict
            X_scaled = self.scalers[student_id].transform(features)
            predicted_minutes = self.models[student_id].predict(X_scaled)[0]
            
            # Apply emergency factor
            if emergency_active:
                predicted_minutes *= 1.3
            
            # Confidence
            confidence = min(pattern.get('sample_count', 0) / 10, 1.0) * 100
            
            return {
                'eta_minutes': max(1, round(predicted_minutes)),
                'eta_seconds': max(60, round(predicted_minutes * 60)),
                'eta_range_min': max(1, round(predicted_minutes * 0.85)),
                'eta_range_max': round(predicted_minutes * 1.15),
                'confidence_percent': round(confidence),
                'current_speed_kmh': round(current_speed, 1),
                'distance_remaining_km': round(distance, 2),
                'traffic_status': self._traffic_status(traffic_factor),
                'emergency_active': emergency_active,
                'estimated_arrival_time': self._arrival_time(predicted_minutes),
                'model_type': 'trained'
            }
        else:
            # Fallback prediction
            if current_speed > 0:
                base_time = (distance / current_speed) * 60
            else:
                base_time = (distance / 20) * 60
            
            # Apply peak hour factor
            if is_peak_hour:
                base_time *= 1.3
            
            # Apply emergency factor
            if emergency_active:
                base_time *= 1.3
            
            return {
                'eta_minutes': max(1, round(base_time)),
                'eta_seconds': max(60, round(base_time * 60)),
                'eta_range_min': max(1, round(base_time * 0.7)),
                'eta_range_max': round(base_time * 1.5),
                'confidence_percent': 50,
                'current_speed_kmh': round(current_speed, 1),
                'distance_remaining_km': round(distance, 2),
                'traffic_status': 'Normal Flow',
                'emergency_active': emergency_active,
                'estimated_arrival_time': self._arrival_time(base_time),
                'model_type': 'fallback'
            }
    
    def _traffic_status(self, factor):
        """Convert traffic factor to status"""
        if factor >= 2.5:
            return "Heavy Traffic"
        elif factor >= 2.0:
            return "Moderate Traffic"
        elif factor >= 1.5:
            return "Light Traffic"
        else:
            return "Normal Flow"
    
    def _arrival_time(self, minutes):
        """Calculate arrival time"""
        arrival = datetime.now() + timedelta(minutes=minutes)
        return arrival.strftime("%H:%M:%S")

# Initialize predictor
predictor = ETAPredictorService()
predictor.load_model()

# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'students_trained': len(predictor.models)
    })

@app.route('/api/predict-eta', methods=['POST'])
def predict_eta():
    """
    Main prediction endpoint
    Called by Node.js backend
    """
    try:
        data = request.json
        
        # Validate required fields
        required = ['student_id', 'bus_location', 'student_destination']
        for field in required:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract data
        student_id = data['student_id']
        bus_location = data['bus_location']
        student_destination = data['student_destination']
        gps_history = data.get('gps_history', {})
        emergency_active = data.get('emergency_active', False)
        
        # Make prediction
        result = predictor.predict(
            student_id,
            bus_location,
            student_destination,
            gps_history,
            emergency_active
        )
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple students
    Called by Node.js for all active students
    """
    try:
        data = request.json
        students = data.get('students', [])
        
        predictions = {}
        
        for student_data in students:
            student_id = student_data['student_id']
            
            result = predictor.predict(
                student_id,
                student_data['bus_location'],
                student_data['student_destination'],
                student_data.get('gps_history', {}),
                student_data.get('emergency_active', False)
            )
            
            predictions[student_id] = result
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """
    Retrain model with new data
    Called periodically by Node.js
    """
    try:
        data = request.json
        
        # Here you would retrain the model with new trip data
        # For now, just reload the existing model
        predictor.load_model()
        
        return jsonify({
            'success': True,
            'message': 'Model reloaded successfully',
            'students_trained': len(predictor.models)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'success': True,
        'model_loaded': predictor.model_loaded,
        'students_trained': len(predictor.models),
        'student_patterns': {
            student_id: {
                'avg_time': pattern.get('avg_time'),
                'sample_count': pattern.get('sample_count')
            }
            for student_id, pattern in predictor.student_patterns.items()
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)