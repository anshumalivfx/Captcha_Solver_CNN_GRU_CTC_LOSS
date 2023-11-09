# Captcha Solver using CNN, GRU and CTC Loss

This repository contains code for a Captcha Image Recognition system. The system is designed to recognize characters in Captcha images.

## Table of Contents

- [Captcha Solver using CNN, GRU and CTC Loss](#captcha-solver-using-cnn-gru-and-ctc-loss)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Model](#model)
  - [Decoding Predictions](#decoding-predictions)
  - [Contributing](#contributing)
  - [License](#license)

---

## Introduction

Captcha (Completely Automated Public Turing test to tell Computers and Humans Apart) is a security mechanism used to determine whether the user is a human or a computer program. This project focuses on recognizing characters in Captcha images.

## Installation

To run this project, you will need Python and several libraries. You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

The project is organized into several Python files:

- `config.py`: Configuration settings for the project.
- `dataset.py`: Defines the dataset and data augmentation.
- `engine.py`: Contains training and evaluation functions.
- `model.py`: Defines the Captcha recognition model.
- `train.py`: The main script for training the model.
- `README.md`: This documentation file.

## Usage

This section describes how to use the code for Captcha recognition.

## Configuration

The `config.py` file contains configuration settings for the project. You can customize these settings, such as image dimensions, batch size, and more, to suit your needs.

## Training

To train the Captcha recognition model, use the `train.py` script. It loads the dataset, preprocesses the data, trains the model, and saves the trained model to a specified path.

## Model

The Captcha recognition model is defined in the `model.py` file. It uses convolutional neural networks (CNNs) and a gated recurrent unit (GRU) for sequence recognition.

## Decoding Predictions

The `decode_prediction` function in `train.py` is used to decode the model's predictions. It takes the model's output, processes it, and returns the decoded Captcha text.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create an issue or a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

This README provides an overview of the project and how to use it. For more detailed information, please refer to the code and comments in the individual Python files.