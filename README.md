# License Plate Recognition System

![image](https://github.com/Sarthak-10/License_Plate_Recognition/assets/55259635/2d6a48c7-64ae-426a-8780-7be86aa9db1e)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Colab_demo.ipynb)

:star:**Please star the repository if you happen to like the project**

This repository contains a Python-based system for license plate recognition (LPR) using image processing and machine learning techniques. The system employs YOLO (You Only Look Once) object detection for identifying license plates in images or videos. It utilizes Tesseract OCR, OpenCV, and various image processing algorithms to extract and decode license plate information.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Demo](#running-the-demo)
- [Contributing](#contributing)
- [License](#license)

## Overview

The system consists of multiple Python scripts:

- `main.py`: Entry point of the application that orchestrates the recognition process.
- `utils/helper.py`: Contains helper functions for color analysis, image processing, and license plate validation.
- `utils/abbr.py`: Stores dictionaries mapping colors and state abbreviations.
- `utils/colorthief.py`: Utilizes color analysis techniques to determine the dominant colors in an image.
- `utils/lp_validation_rules.py`: Holds functions for validating and decoding license plate information.
- `predict.py`: Uses YOLO for object detection, specifically for detecting license plates in videos.
- `train.py`: Trains the YOLO model on labeled data.

## Dependencies

The project relies on the following libraries and frameworks:

- OpenCV
- PyTesseract
- Matplotlib
- NumPy
- Ultralytics (YOLO)
- Torch

Ensure that these dependencies are installed before running the system.

## Usage

### Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/Sarthak-10/License_Plate_Recognition.git
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```
3. Install the additional dependencies:

```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo apt-get install -y libleptonica-dev
sudo apt-get update -y
sudo apt-get install automake
sudo apt-get install -y pkg-config
sudo apt-get install -y libsdl-pango-dev
sudo apt-get install -y libicu-dev
sudo apt-get install -y libcairo2-dev
sudo apt-get install bc
```
### Running the Demo
1. Ensure you have the necessary input data:
   - Sample cropped license plate images should be in the `cropped_lp_samples` folder.

2. Run the system by executing `main.py`:

```bash
python main.py 
```

The project is able to execute specific functionalities, such as:

- Recognizing license plates from cropped images
- Analyzing dominant colors in license plate images
- Validating and decoding license plate information
- Training and using the YOLO model for object detection

## Flow of the Project

In the project, we first train a yolov5 model to detect license plates from traffic surveillance images from a custom made traffic dataset. Then we use the model to detect license plates in a demo video. The cropped images of the detected license plates are stored in a folder. We then perform digital image processing techniques like deblurring, sharpening, etc on the cropped images of the license plate images. Finally we use Google's tesseract engine for OCR. We pass the processed images via the tesseract library to obtain the license plate numbers. We further post-process, validate and analyze the obtained license plate numbers to gather the information about the particular class and regions the vehicle belongs.
  
## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
