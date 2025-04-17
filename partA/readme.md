# Part A: Training from scratch
####  Setup
 Change the direcotry:
   ```bash
   cd partA
   ```
### `model.py`
The model is designed for multi-class image classification tasks and is tailored to work with the iNaturalist dataset (10 classes). It provides flexibility regarding layer configuration, activation functions, and regularization techniques.
- Modular CNN architecture with 5 convolutional blocks
- Customizable filter configuration, kernel size, and activation functions
- Optional Batch Normalization and Dropout
- Automatically handles flattening based on input image size
- Designed to work with inputs of shape (3, 256, 256) (iNaturalist dataset standard)

### `sweep_config.py`
The sweep helps in identifying the best-performing combination of model architecture and training strategy by optimizing for validation accuracy.
#### Sweep Overview

| Parameter        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `method`         | Optimization strategy used in the sweep. Set to `"bayes"` for Bayesian optimization. |
| `metric.name`    | The metric used to evaluate performance. Set to `"val_acc"` (validation accuracy). |
| `metric.goal`    | Objective of the sweep. Set to `"maximize"` to find configurations that yield the highest accuracy. |


#### Tunable Hyperparameters

| Parameter           | Type        | Description                                                                 |
|---------------------|-------------|-----------------------------------------------------------------------------|
| `filters_per_layer` | list        | Controls the number of filters in each convolutional block (5 blocks total). |
| `activation`        | categorical | Activation function used in the model. Options: `ReLU`, `GELU`, `SiLU`, `Mish`. |
| `use_batchnorm`     | boolean     | Whether to use Batch Normalization after each convolutional block.          |
| `dropout_rate`      | float       | Dropout probability used after convolutional and dense layers.              |
| `dense_units`       | int         | Number of units in the fully connected dense layer before output.           |
| `augmentation`      | boolean     | Whether to apply data augmentation during training.                         |
| `batch_size`        | int         | Number of samples per training batch. Options: `32`, `64`, `128`.           |
| `learning_rate`     | float       | Learning rate used by the Adam optimizer.                                   |
| `epochs`            | int         | Number of epochs to train the model. Options: `20`, `15`, `10`.               |

### `train.py`
- Run the Script:
   ```bash
   python train.py
   ```
This script trains a configurable Convolutional Neural Network (CNN) on the iNaturalist 12K dataset using PyTorch. It integrates Weights & Biases (wandb) for experiment tracking and supports sweep-based hyperparameter tuning. It also saves the best-performing model based on validation accuracy.
- Leverages the modular CNN model from `Question 1: model.py`
- Uses wandb sweeps to perform hyperparameter optimization
- Implements optional data augmentation, batch normalization, and dropout
- Tracks training/validation loss and accuracy across epochs
- Saves the best model (based on validation accuracy) to `best_model.pth`

### `test_model.py`
- Run the Script:
   ```bash
   python test_model.py
   ```
The `test_model.py` script is used to evaluate the performance of the best-trained CNN model on the **test split** of the iNaturalist_12K dataset. This script also logs final metrics and predictions to **Weights & Biases (wandb)** for visualization and reporting.


- Loads and applies the best configuration from `best_config.py`
- Evaluates test accuracy using the saved model (`model_path`)
- Visualizes predictions on a 10x3 image grid with true vs. predicted labels
- Logs:
  - Test accuracy
  - Sample prediction grid (image panel) `10*3`
