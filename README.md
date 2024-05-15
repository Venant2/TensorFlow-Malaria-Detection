Project Description

This project focuses on developing a deep learning model for automated malaria diagnosis by classifying microscopic blood smear images into parasitized (infected) and uninfected categories.

Dataset:

The model was trained and validated using a dataset of cell images. The dataset was curated to include a balanced representation of both parasitized and uninfected cells, ensuring the model learns to differentiate between the two classes effectively.

Model Architecture:

Due to the computational constraints of working with a CPU, a relatively simple CNN architecture was chosen, consisting of:

Two convolutional layers (16 and 32 filters) with 3x3 kernels and ReLU activation, each followed by max-pooling (2x2).
Dropout layers (0.2 and 0.3) to mitigate overfitting after each convolutional block.
A flattening layer followed by a dense layer (64 units, ReLU activation) with dropout (0.5).
A final dense layer (1 unit, sigmoid activation) for binary classification.
While a deeper network with more layers could potentially yield better performance, the current architecture strikes a balance between accuracy and computational efficiency on a CPU.

Training:

The model was trained for 5 epochs using the Adam optimizer and binary cross-entropy loss with a batch size of 16. Early stopping and model checkpointing were employed to prevent overfitting and save the best-performing model based on validation loss.

Results:

The final model achieved an accuracy of 93.58% and a loss of 0.1843 on the validation set, demonstrating its effectiveness in distinguishing between parasitized and uninfected cells. Detailed training progress is visualized in model_results.pdf.

The model consistently improved accuracy across epochs, with validation accuracy closely mirroring training accuracy.
The validation loss progressively decreased, suggesting good generalization capabilities.
Potential for Further Improvement:

Given sufficient computational resources (e.g., access to a GPU), the model's performance could potentially be enhanced through:

Deeper Network Architecture: Adding more convolutional and/or dense layers could enable the model to learn more complex features and representations.
Hyperparameter Optimization: Fine-tuning hyperparameters like learning rate, dropout rates, and batch size could further optimize the model.
Additional Regularization Techniques: Techniques like batch normalization could be explored to improve generalization and prevent overfitting.
Conclusion:

Despite working with a CPU-only environment, this project demonstrates the feasibility of developing an accurate and efficient malaria cell classification model. The achieved results showcase the potential of deep learning in aiding medical diagnosis, even with limited computational resources. Further research with more powerful hardware could unlock even greater potential for this approach.
Usage:

Clone this repository.
Install required dependencies (pip install tensorflow keras).
Prepare your dataset in the following structure:
cell_images/
    Parasitized/
    Uninfected/
Run the Python script to train and evaluate the model.
Future Work:

Experiment with different CNN architectures and hyperparameters.
Implement GPU acceleration for faster training and inference.
Explore transfer learning with pre-trained models.
