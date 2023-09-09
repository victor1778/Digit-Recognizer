# Digit Recognition from Scratch

This repository contains Python code for training a neural network from scratch using NumPy. The neural network is trained on the MNIST dataset for handwritten digit recognition.

## Overview

- [Description](#description)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Description

This project implements a simple feedforward neural network with one hidden layer to classify handwritten digits. The code includes functionalities for data preprocessing, model initialization, forward and backward propagation, gradient descent optimization, and early stopping based on development set accuracy.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- NumPy
- pandas
- matplotlib

You can install these dependencies using `pip`:

```bash
pip install numpy pandas matplotlib
```
## Usage
1. Clone this repository to your local machine:
	```bash
	git clone https://github.com/victor1778/Digit-Recognizer.git
	```
2. Navigate to the project directory 
	```bash
	cd Digit-Recognizer
	```
3.  Download the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and save it as `mnist_train.csv` in the project directory.
    
4.  Run the training script:
	```bash
	python train_neural_network.py
	```
5.  Run the test script:
	```bash
	python test_model.py
	```
	
## Training

-   The training script `train_neural_network.py` loads the MNIST dataset, preprocesses it, and trains a neural network with specified hyperparameters.
    
-   You can customize the hyperparameters such as hidden layer size, learning rate, number of iterations, and early stopping patience in the script.
    

## Results

-   The trained model will be saved as `trained_model.pkl` in the project directory.
    
-   Training loss and development accuracy graphs will be displayed during training.

    ![Training Data](https://github.com/victor1778/Digit-Recognizer/blob/f84a0da795613ded8329206ec95b948951987467/img/Training_Data.png)

- The test script will load the trained model, preprocess the test data, make predictions, and compute the accuracy on the test dataset.

-   Sample images from the test dataset, along with their true and predicted labels, will be displayed.

    | ![Figure 1](https://github.com/victor1778/Digit-Recognizer/blob/f84a0da795613ded8329206ec95b948951987467/img/Figure_1.png) | ![Figure 2](https://github.com/victor1778/Digit-Recognizer/blob/f84a0da795613ded8329206ec95b948951987467/img/Figure_2.png) |
    |--|--|
    | ![Figure 3](https://github.com/victor1778/Digit-Recognizer/blob/f84a0da795613ded8329206ec95b948951987467/img/Figure_4.png) | ![Figure 4](https://github.com/victor1778/Digit-Recognizer/blob/f84a0da795613ded8329206ec95b948951987467/img/Figure_5.png) |

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/victor1778/Digit-Recognizer/main/LICENSE).
