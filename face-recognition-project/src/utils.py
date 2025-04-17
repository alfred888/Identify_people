def display_image(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def evaluate_model(model, test_data, test_labels):
    from sklearn.metrics import classification_report, confusion_matrix
    predictions = model.predict(test_data)
    predicted_classes = predictions.argmax(axis=1)
    print(confusion_matrix(test_labels, predicted_classes))
    print(classification_report(test_labels, predicted_classes))

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)