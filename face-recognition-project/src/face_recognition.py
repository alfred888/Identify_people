import cv2
import numpy as np
import pandas as pd
from keras.models import load_model

class FaceRecognition:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)
        self.labels = pd.read_csv(labels_path)
        self.label_dict = {row['id']: row['name'] for index, row in self.labels.iterrows()}

    def preprocess_image(self, image):
        image = cv2.resize(image, (224, 224))  # Resize to match model input
        image = image.astype('float32') / 255.0  # Normalize the image
        return np.expand_dims(image, axis=0)  # Add batch dimension

    def recognize_face(self, image_path):
        image = cv2.imread(image_path)
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        return self.label_dict.get(predicted_class, "Unknown")

if __name__ == "__main__":
    model_path = '../models/face_recognition_model.h5'
    labels_path = '../data/labels.csv'
    face_recognition = FaceRecognition(model_path, labels_path)

    test_image_path = 'path_to_test_image.jpg'  # Replace with the path to your test image
    recognized_name = face_recognition.recognize_face(test_image_path)
    print(f'Recognized: {recognized_name}')