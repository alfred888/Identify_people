import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def save_processed_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename, img in images:
        processed_img = preprocess_image(img)
        cv2.imwrite(os.path.join(output_folder, filename), processed_img * 255)  # Convert back to [0, 255]

def load_labels(csv_file):
    return pd.read_csv(csv_file)

def split_data(images, labels, test_size=0.2):
    return train_test_split(images, labels, test_size=test_size, random_state=42)

if __name__ == "__main__":
    raw_images_folder = '../data/raw_images'
    processed_images_folder = '../data/processed_images'
    labels_file = '../data/labels.csv'

    images = load_images_from_folder(raw_images_folder)
    labels = load_labels(labels_file)

    save_processed_images(images, processed_images_folder)

    # Further processing can be done here, such as splitting data for training and validation
    # images_train, images_val, labels_train, labels_val = split_data(images, labels)