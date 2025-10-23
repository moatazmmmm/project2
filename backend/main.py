import os
import io
import time
import logging
import sqlite3
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_401_UNAUTHORIZED
from tensorflow.keras.models import load_model
from .model_loader import get_model
from .preprocessing import preprocess_image, CLASS_NAMES
from pydantic import BaseModel

# Config
API_KEY = os.environ.get('API_KEY', 'testkey')
MODEL_PATH = os.environ.get('MODEL_PATH', 'best_model.h5')
RATE_LIMIT = int(os.environ.get('RATE_LIMIT_PER_MIN', '60'))  # requests per minute per IP

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='CIFAR-10 Prediction API', version='1.0')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Simple in-memory rate limiter (bonus)
RATE_STATE = {}

def rate_limiter(ip: str):
    window = 60  # seconds
    now = time.time()
    calls = RATE_STATE.get(ip, [])
    # remove old
    calls = [t for t in calls if now - t < window]
    if len(calls) >= RATE_LIMIT:
        return False
    calls.append(now)
    RATE_STATE[ip] = calls
    return True

# Simple API key dependency
async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail='Invalid API Key')

# DB logging
DB_PATH = base = os.path.join(os.path.dirname(__file__), 'predictions.db')
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  client_ip TEXT,
                  filename TEXT,
                  predicted_class TEXT,
                  confidence REAL,
                  details TEXT)''')
    conn.commit()
    conn.close()

@app.on_event('startup')
async def startup_event():
    logger.info('Starting up -> loading model')
    init_db()
    # load model via model_loader (cached)
    get_model()
    logger.info('Model loaded (or attempted).')

@app.middleware('http')
async def log_requests(request: Request, call_next):
    client_ip = request.client.host if request.client else 'unknown'
    logger.info(f'Incoming request {request.method} {request.url} from {client_ip}')
    start = time.time()
    response = await call_next(request)
    process_time = (time.time() - start) * 1000
    logger.info(f'Completed {request.method} {request.url} in {process_time:.2f}ms status={response.status_code}')
    return response

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    predictions: dict

@app.get('/health', tags=['Health'])
async def health():
    model = get_model()
    model_loaded = model is not None
    return {'status': 'healthy', 'model_loaded': model_loaded}

def log_prediction(client_ip, filename, predicted_class, confidence, details):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, client_ip, filename, predicted_class, confidence, details) VALUES (?,?,?,?,?,?)',
              (time.strftime('%Y-%m-%d %H:%M:%S'), client_ip, filename, predicted_class, float(confidence), details))
    conn.commit()
    conn.close()

@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'], dependencies=[Depends(verify_api_key)])
async def predict(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host if request.client else 'unknown'
    if not rate_limiter(client_ip):
        raise HTTPException(status_code=429, detail='Rate limit exceeded')
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(status_code=400, detail='Only JPEG and PNG images are supported')
    contents = await file.read()
    try:
        img = preprocess_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error preprocessing image: {e}')
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    preds = model.predict(img)
    probs = preds[0].tolist()
    class_probs = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    response = PredictionResponse(
        class_name=CLASS_NAMES[top_idx],
        confidence=float(probs[top_idx]),
        predictions=class_probs
    )
    # log to DB
    log_prediction(client_ip, getattr(file, 'filename', 'upload'), response.class_name, response.confidence, str(class_probs))
    return response

@app.post('/predict/batch', tags=['Prediction'], dependencies=[Depends(verify_api_key)])
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    client_ip = request.client.host if request.client else 'unknown'
    if not rate_limiter(client_ip):
        raise HTTPException(status_code=429, detail='Rate limit exceeded')
    results = []
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    for file in files:
        if file.content_type not in ('image/jpeg', 'image/png'):
            results.append({'filename': getattr(file, 'filename', 'upload'), 'error': 'Unsupported file type'})
            continue
        contents = await file.read()
        try:
            img = preprocess_image(contents)
        except Exception as e:
            results.append({'filename': getattr(file, 'filename', 'upload'), 'error': f'Preprocess error: {e}'})
            continue
        preds = model.predict(img)
        probs = preds[0].tolist()
        class_probs = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
        top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        res = {'filename': getattr(file, 'filename', 'upload'), 'class': CLASS_NAMES[top_idx], 'confidence': float(probs[top_idx]), 'predictions': class_probs}
        results.append(res)
        log_prediction(client_ip, getattr(file, 'filename', 'upload'), res['class'], res['confidence'], str(class_probs))
    return JSONResponse(content={'results': results})
