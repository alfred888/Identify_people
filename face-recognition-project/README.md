# Face Recognition Project

This project implements a face recognition system using deep learning techniques. The goal is to train a model that can recognize specific individuals based on their facial images.

## Project Structure

- **data/**
  - **raw_images/**: This folder contains the original facial image data.
  - **processed_images/**: This folder stores the processed facial image data.
  - **labels.csv**: This file contains the label information for each image, used for training the model.

- **models/**
  - **face_recognition_model.h5**: This file is the trained face recognition model, saved in HDF5 format.

- **src/**
  - **data_preprocessing.py**: This file includes code for data preprocessing, including image loading, resizing, and augmentation.
  - **model_training.py**: This file contains the model training code, defining the model architecture and training it using the processed data.
  - **face_recognition.py**: This file implements the face recognition functionality, using the trained model to recognize new images.
  - **utils.py**: This file includes utility functions, such as image display and result evaluation.

- **requirements.txt**: This file lists the required Python libraries and their versions for the project.

- **README.md**: This file contains the documentation and usage instructions for the project.

- **.gitignore**: This file specifies files and folders to be ignored in version control.

## Installation

To set up the project, clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your raw facial images in the `data/raw_images/` directory.
2. Run the `src/data_preprocessing.py` script to preprocess the images and save them in the `data/processed_images/` directory.
3. Use the `src/model_training.py` script to train the face recognition model.
4. After training, the model will be saved in the `models/` directory.
5. To recognize faces in new images, use the `src/face_recognition.py` script.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.