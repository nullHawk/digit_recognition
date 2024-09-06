# Digit Recognizer using Feed-Forward Neural Network

## Overview

This project implements a digit recognition system using a Feed-Forward Neural Network (FFNN). The model is trained to recognize handwritten digits from the MNIST dataset and is deployed using a web interface.

## Features

- **Digit Recognition**: Classify handwritten digits from the MNIST dataset.
- **Web Interface**: A user-friendly interface built with Gradio to draw and recognize digits in real-time.
- **Model Deployment**: Easily deploy and interact with the model.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- torchvision
- Gradio
- numpy
- Pillow

You can install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt

```

## Installation

1. **Clone the Repository:**
    
    ```bash
    git clone https://github.com/nullHawk/digit_recognition_ffnn.git
    
    ```
    
2. **Navigate to the Project Directory:**
    
    ```bash
    cd digit_recognition_ffnn
    
    ```
    
3. **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    
    ```
    

## Usage

1. **Prepare the Model:**
    
    Ensure you have the trained model file (`model.pt`) saved in the `model` directory.
    
2. **Run the Web Interface:**
    
    ```bash
    python app.py
    
    ```
    
3. **Open Your Browser:**
    
    Access the web interface at `http://localhost:7860` to draw digits and get predictions.
    

## Example

Here’s an example of how to use the web interface:

1. Draw a digit on the canvas.
2. Click on "Predict" to see the recognized digit.

## Directory Structure

```bash
bashCopy code
repository/
│
├── app.py            # Main script to run the Gradio interface
├── model/
│   └── model.pt      # Trained model file
├── requirements.txt  # List of dependencies
└── README.md         # This file

```

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For significant changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- **MNIST Dataset**: Provided by Yann LeCun and the NYU.
- **PyTorch**: Open-source machine learning library used for model development.
- **Gradio**: Framework used for creating the web interface.
