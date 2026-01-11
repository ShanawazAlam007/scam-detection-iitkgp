import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from prediction_pipeline import ScamDetector

# Use absolute paths for model files for robustness in any environment
# __file__ gives the path of the current script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizer.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'metadata_scaler.pkl')

# Load the model once at startup
print("Loading scam detection model...")
try:
    detector = ScamDetector(MODEL_PATH, TOKENIZER_PATH, SCALER_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load model. Error: {e}")
    detector = None

# Initialize Flask app
app = Flask(__name__)

# Enable CORS with explicit configuration to avoid CORS errors
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins (change to specific domain in production)
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Configuration
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TIMEOUT'] = 300  # 5 minutes timeout
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max request size
app.config['JSON_SORT_KEYS'] = False

# Constants for validation
MAX_TEXT_LENGTH = 50000  # Maximum characters in text input
MIN_TEXT_LENGTH = 1  # Minimum characters in text input

@app.route("/", methods=["GET"])
def root():
    """Root endpoint to prevent 404."""
    return jsonify({"status": "ok", "message": "Scam Detection API is running."})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with model status."""
    model_status = "loaded" if detector is not None else "not_loaded"
    return jsonify({
        "status": "ok",
        "model_status": model_status,
        "version": "1.0"
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Prediction endpoint with CORS preflight support."""
    # Handle OPTIONS request for CORS preflight
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response, 200
    
    if not detector:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 500

    try:
        # Parse JSON with error handling
        try:
            data = request.get_json(force=True)
        except Exception as json_error:
            return jsonify({"error": "Invalid JSON format."}), 400
        
        # Validate required fields
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input: 'text' field is required."}), 400

        text = data["text"]
        
        # Validate text input
        if not isinstance(text, str):
            return jsonify({"error": "'text' field must be a string."}), 400
        
        if len(text.strip()) < MIN_TEXT_LENGTH:
            return jsonify({"error": "'text' field cannot be empty."}), 400
        
        if len(text) > MAX_TEXT_LENGTH:
            return jsonify({"error": f"'text' field exceeds maximum length of {MAX_TEXT_LENGTH} characters."}), 400
        
        metadata = data.get("metadata", {}) # Safely get metadata, default to empty dict
        
        # Validate metadata if provided
        if metadata and not isinstance(metadata, dict):
            return jsonify({"error": "'metadata' field must be an object."}), 400

        # The predict function now returns a dictionary
        result = detector.predict(text, metadata)

        # Extract values for logging
        prediction_label = "scam" if result.get('prediction') == 1 else "safe"
        source = result.get('source', 'N/A')
        confidence = result.get('confidence')
        
        # Ensure confidence is JSON serializable for logging
        if isinstance(confidence, np.floating):
            confidence = float(confidence)

        # Log details to terminal for debugging
        print("--- New Prediction ---")
        print(f"Input Text: {text}")
        print(f"Decision Source: {source}")
        print(f"Final Prediction: {prediction_label}")
        print(f"Confidence: {confidence:.4f}" if confidence is not None else "Confidence: N/A")
        print("----------------------")

        # Return the response with ONLY the prediction key, as required
        response = jsonify({"prediction": prediction_label})
        # Add CORS headers explicitly to ensure compatibility
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    except ValueError as ve:
        # Handle specific validation errors
        print(f"Validation error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Log the full error for debugging but return a generic message
        print(f"ERROR: An unhandled exception occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during prediction."}), 500

# Global error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle payload too large errors."""
    return jsonify({"error": "Request payload is too large. Maximum size is 10MB."}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found."}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors."""
    return jsonify({"error": "Method not allowed for this endpoint."}), 405

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors."""
    return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    if not detector:
        print("FATAL: Backend not starting because model failed to load.")
        exit(1)
    else:
        print("="*60)
        print("Starting Flask backend for local testing on http://0.0.0.0:5000")
        print("Backend will be accessible from any network interface")
        print("CORS enabled for all origins")
        print(f"Max text length: {MAX_TEXT_LENGTH} characters")
        print(f"Max request size: 10MB")
        print("="*60)
        # Use 0.0.0.0 to allow connections from other devices/frontend
        # Disable debug mode in production, use threaded=True for better performance
        try:
            app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"\nFATAL: Server crashed with error: {e}")
            exit(1)
