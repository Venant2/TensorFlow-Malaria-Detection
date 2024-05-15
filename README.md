Malaria Cell Classification with CNN (CPU Optimized)

This project demonstrates a convolutional neural network (CNN) model for automated malaria diagnosis by classifying microscopic blood smear images into parasitized (infected) and uninfected categories. Despite being optimized for CPU environments, the model achieves high accuracy, showcasing the potential of deep learning in medical image analysis even with limited computational resources.

Key Features

Optimized for CPU: Specifically designed for CPU environments, making it accessible to a wider range of users and platforms.
High Accuracy: Achieves a validation accuracy of 93.58% on the test dataset, demonstrating strong performance in distinguishing between infected and uninfected cells.
Scalable Architecture: Utilizes a streamlined CNN architecture, easily adaptable for further performance enhancements with GPU acceleration.
Transparent Results: Includes a detailed PDF report (model_results.pdf) visualizing the model's training progress and performance metrics.
Dataset

A carefully curated dataset of microscopic blood smear images was used, ensuring a balanced representation of both parasitized and uninfected cells for effective model training and evaluation.

Methods

TensorFlow and Keras: Leveraged the powerful deep learning frameworks of TensorFlow and Keras for model implementation.
ImageDataGenerator: Employed ImageDataGenerator for efficient data preprocessing, including image resizing and normalization.
Model Architecture: Designed a lightweight yet effective CNN architecture, comprising convolutional, pooling, and dense layers, tailored for optimal performance on CPUs.
Future Work

With access to GPU resources, the model's performance could be further enhanced through:

Deeper network architectures
Hyperparameter optimization
Implementation of additional regularization techniques
Usage

Clone this repository.
Install required dependencies (pip install tensorflow keras).
Prepare your dataset in the following structure:
cell_images/
    Parasitized/
    Uninfected/
Run the Python script to train and evaluate the model.
Feel free to explore and build upon this project, leveraging the insights gained to advance malaria diagnosis through deep learning.
