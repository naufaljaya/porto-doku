# Object Detection Tensorflow

This code is a deep learning project that involves building and training a YOLO (You Only Look Once) object detection model using the EfficientNetB1 architecture. The project includes data preprocessing, model creation, training, and testing for object detection on images. Here's an overview of the code:

## ⭐ Credit

Folefac Martins from Neuralearn.ai

### Data Setup

1.  The Kaggle API is installed and configured to download the "Pascal VOC 2012" dataset.
2.  Images and annotations from the dataset are moved to appropriate folders for training and validation.

### Data Preprocessing

1.  Functions are defined for processing and augmenting XML annotations.
2.  Functions are defined for loading and preprocessing dataset samples.
3.  Augmentations are applied using the Albumentations library.
4.  The training and validation datasets are created and preprocessed.

### Model Definition

1.  The EfficientNetB1 model is imported from TensorFlow's applications module and configured for transfer learning.
2.  A custom model is defined using TensorFlow's Sequential API. It includes convolutional layers, batch normalization, leaky ReLU activations, dense layers, and reshaping operations.

### Loss Function

A custom YOLO loss function is defined that includes different components for object detection, no-object prediction, class prediction, and bounding box prediction. The loss function calculates and combines these components.

### Training

1.  Callbacks for learning rate scheduling and model checkpointing are defined.
2.  The model is compiled with the custom YOLO loss and the Adam optimizer.
3.  The model is trained using the defined training and validation datasets, along with the specified number of epochs.

### Model Testing

A function is defined to perform object detection on test images using the trained model. The function uses non-maximum suppression to filter detected objects and draws bounding boxes with class labels on the images.

### Testing the Model

The trained model is loaded from the checkpoint, and the defined function is used to perform object detection on a set of test images from the "COCO" dataset. Detected objects are highlighted with bounding boxes and class labels, and the resulting images are saved in an "outputs" folder.
