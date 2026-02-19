from src.io_utils import load_pickle
from keras.models import load_model
from src.config import MODEL_PATH, LABEL_ENCODER_FILEPATH, ONE_HOT_ENCODER_FILEPATH, STANDARD_SCALER_FILEPATH

class InferenceModel:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.label_encoder = load_pickle(LABEL_ENCODER_FILEPATH)
        self.one_hot_encoder = load_pickle(ONE_HOT_ENCODER_FILEPATH)
        self.standard_scaler = load_pickle(STANDARD_SCALER_FILEPATH)
