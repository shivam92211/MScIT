# Flask ML App: Machine Learning Model Deployment with Docker

This project demonstrates how to train a simple machine learning model, save it, create a Flask API to serve the model, and Dockerize the application for deployment.

---

## Features

- **Machine Learning**: Trained a Random Forest Classifier using the Iris dataset.
- **Model Persistence**: Saved the trained model using Python's `pickle` module.
- **Flask API**: Created an API endpoint to make predictions using the saved model.
- **Dockerization**: Packaged the application into a Docker container for easy deployment.

---

## Requirements

- Python 3.9 or later
- Docker (for containerization)
- Required Python libraries:
  - Flask
  - scikit-learn

Install Python dependencies using:

```bash
pip install flask scikit-learn
```

---

## Setup and Usage

### 1. Train and Save the Model

Run the script `save_model.py` to train the model and save it as `model.pkl`:

```bash
python save_model.py
```

### 2. Start the Flask API

Run the Flask app:

```bash
python app.py
```

The app will start at `http://0.0.0.0:5000`.

### 3. Test the API

You can test the API using a tool like `curl` or Postman.

#### Example Request:

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

#### Example Response:

```json
{"prediction": 0}
```

---

## Dockerization

### 1. Build the Docker Image

Create a Docker image using the provided `Dockerfile`:

```bash
docker build -t flask-ml-app .
```

### 2. Run the Docker Container

Run the container and map it to port 5000:

```bash
docker run -d -p 5000:5000 flask-ml-app
```

### 3. Access the API

The API will be available at `http://localhost:5000/predict`.

---

## Project Structure

```plaintext
.
├── app.py              # Flask app for serving the model
├── save_model.py       # Script to train and save the ML model
├── model.pkl           # Saved Random Forest model
├── Dockerfile          # Docker configuration file
├── requirements.txt    # Python dependencies
```

---

## How It Works

1. **Model Training**:
   - The `save_model.py` script trains a Random Forest Classifier on the Iris dataset and saves the model to `model.pkl`.

2. **Flask API**:
   - The `app.py` script serves the saved model through a Flask API with a single `/predict` endpoint.

3. **Dockerization**:
   - The `Dockerfile` defines the environment, installs dependencies, and runs the Flask app inside a container.

---

## Notes

- The `features` input must be a list of 4 numerical values (corresponding to the Iris dataset features).
- Modify the model or API logic to adapt to different datasets or use cases.

---

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code.