import os
from tensorflow.keras.models import load_model
import threading
from typing import Optional
MODEL_PATH = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), 'best_model.h5'))
_model = None
_lock = threading.Lock()

def _load():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            print(f'Warning: model file {MODEL_PATH} not found. Please place your trained model at that path.')
            return None
        try:
            _model = load_model(MODEL_PATH)
            print('Model loaded successfully from', MODEL_PATH)
        except Exception as e:
            print('Error loading model:', e)
            _model = None
    return _model

def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _load()
    return _model
