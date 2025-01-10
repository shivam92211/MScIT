# app.py

from flask import Flask, request, jsonify
import pickle

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get JSON data from the request
    if not data or "features" not in data:
        return jsonify({"error": "Invalid input. Expected JSON with 'features' key."}), 400
    
    features = data["features"]
    
    # Ensure the input is in the correct format
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "Input must be a list of 4 numeric values."}), 400

    try:
        prediction = model.predict([features])
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
