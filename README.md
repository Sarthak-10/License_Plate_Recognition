# License Plate Recognition System

This repository contains a Python-based system for license plate recognition (LPR) using image processing and machine learning techniques. The system employs YOLO (You Only Look Once) object detection for identifying license plates in images or videos. It utilizes Tesseract OCR, OpenCV, and various image processing algorithms to extract and decode license plate information.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the System](#running-the-system)
- [Folder Structure](#folder-structure)
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

### Running the System

1. Ensure you have the necessary input data:
   - Place video files in the `detect/predict` directory.
   - Sample cropped license plate images should be in the `cropped_lp_samples` folder.

2. Run the system by executing `main.py`:

```bash
python main.py
```

Follow the prompts and instructions to execute specific functionalities, such as:

- Recognizing license plates from cropped images
- Analyzing dominant colors in license plate images
- Validating and decoding license plate information
- Training and using the YOLO model for object detection

## Folder Structure

- `detect/predict`: Contains video files for license plate detection.
- `runs`: Stores the outputs generated during model training.
- `cropped_lp_samples`: Holds cropped license plate images for testing and validation.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
