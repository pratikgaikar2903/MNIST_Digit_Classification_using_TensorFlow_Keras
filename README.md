# MNIST Digit Classification using TensorFlow/Keras

This repository contains an implementation of a Multi-Layer Perceptron (MLP) built with Keras and TensorFlow to accurately classify handwritten digits from the MNIST dataset.

## Project Overview
The objective of this project is to build, train, and evaluate a deep feedforward neural network that can recognize handwritten digits (0 through 9). The model is designed using Sequential fully connected layers.

## Dataset
We utilize the classic **MNIST dataset** loaded directly via Keras:
* **Training Set**: 60,000 images
* **Test Set**: 10,000 images
* **Dimensions**: $28 \times 28$ pixels in grayscale

### Preprocessing
To aid gradient descent and speed up convergence, the pixel values are scaled from their original range of `[0, 255]` to `[0.0, 1.0]`:
```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

## Model Architecture
The model is a feedforward neural network structured as follows:
1. **Flatten Layer**: Flattens the $28 \times 28$ input images into a $784$-dimensional 1D vector.
2. **Dense Hidden Layer 1**: 128 units with ReLU activation.
3. **Dense Hidden Layer 2**: 64 units with ReLU activation.
4. **Dense Hidden Layer 3**: 32 units with ReLU activation.
5. **Dense Output Layer**: 10 units with Softmax activation (one for each digit class).

```python
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(10, activation="softmax")
])
```

## Training and Optimization
* **Optimizer**: `Adam`
* **Loss Function**: `Sparse Categorical Crossentropy` (since labels are integers)
* **Metrics**: `Accuracy`
* **Epochs**: 10
* **Batch Size**: 32
* **Validation Split**: 20% of the training dataset

## Evaluation and Results
After training for 10 epochs, the model is evaluated on the unseen test set:
* **Test Loss**: ~`0.119`
* **Test Accuracy**: ~`97.17%`
