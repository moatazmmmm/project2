# CIFAR-10 Prediction API (FastAPI)

## Overview
This backend implements a FastAPI service to serve a trained CIFAR-10 model. It includes:

- `/health` GET health check
- `/predict` POST single image prediction (multipart/form-data)
- `/predict/batch` POST multiple images
- CORS enabled, simple rate limiting, API key auth, and SQLite logging

## Project structure
```
backend/
  ├── main.py
  ├── model_loader.py
  ├── preprocessing.py
  ├── best_model.h5   <-- place your trained Keras model here
  ├── predictions.db  <-- created automatically
  ├── requirements.txt
  └── README.md
```

## Setup
1. Create virtual environment (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Place your trained model file `best_model.h5` inside the `backend/` folder or set `MODEL_PATH` environment variable.
4. (Optional) Set API_KEY and RATE_LIMIT:
   ```bash
   export API_KEY='my-secret-key'
   export RATE_LIMIT_PER_MIN=60
   ```
5. Run the app
   ```bash
   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Usage
- Health:
  ```bash
  curl http://localhost:8000/health
  ```
- Single predict (replace IMAGE.jpg and API key):
  ```bash
  curl -X POST -H "x-api-key: testkey" -F "file=@IMAGE.jpg" http://localhost:8000/predict
  ```

## Notes
- Swagger UI available at `http://localhost:8000/docs`
- Rate limiting implemented in-memory (not suitable for multi-instance production)
- For production use, replace rate limiter with Redis-based limiter and secure your API key management
