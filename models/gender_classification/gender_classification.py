import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from keras.models import load_model

class GenderClassifier():
    MODEL_PATH = "models/gender_classification/gender_classification.model"
    CLASSES = ["man", "woman"]

    def __init__(
        self, model_path=None, classes=None
    ):
        self._classes = classes if classes else GenderClassifier.CLASSES

        model_path = model_path if model_path else GenderClassifier.MODEL_PATH
        self._check_if_file_exists(model_path)
        self._model = load_model(model_path)

    def _check_if_file_exists(self, path):
        with open(path) as f:
            pass

    def _load_image_by_path(self, image_path: str):
        self._check_if_file_exists(image_path)
        image = cv2.imread(image_path)
        return image

    def _preprocess_image(self, image):
        image = cv2.resize(image, (96,96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def _get_label_from_prediction(self, prediction):
        confidence = prediction[0]
        idx = np.argmax(confidence)
        label = self._classes[idx]
        return label

    def predict_by_image_path(self, image_path: str) -> str:
        image = self._load_image_by_path(image_path)
        image = self._preprocess_image(image)
        prediction = self._model.predict(image)
        label = self._get_label_from_prediction(prediction)
        return label

    def get_model(self):
        return self._model
