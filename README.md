# EMT Dataset Evaluation

Code repository for evaluating multiple object trackers (MOT) using the EMT dataset.

## Overview

This repository provides tools and scripts for evaluating tracking algorithms on the EMT (Emergency Medical Team) dataset using the TrackEval framework.

## Prerequisites

- Python 3.7 or higher
- Conda (recommended for environment management)
- Git

## Installation

### 1. TrackEval Setup

First, install the TrackEval framework:

```bash
# Clone TrackEval repository
git clone https://github.com/JonathonLuiten/TrackEval.git

# Create and activate conda environment
conda create -n trackeval python=3.7
conda activate trackeval

# Install TrackEval
cd TrackEval/
pip install matplotlib
pip install -e .
cd ..
```

### 2. Dataset Configuration

Add the EMT dataset support to TrackEval:

1. Navigate to the TrackEval datasets directory
2. Add the `emt_dataset.py` file
3. Update the `__init__.py` file by adding:
```python
from .emt_dataset import EMT2DBox
```

### 3. Tracker Installation

#### ByteTracker

```bash
# Clone ByteTrack repository
git clone https://github.com/ifzhang/ByteTrack.git

# Install dependencies
cd ByteTrack
pip install -r requirements.txt
python setup.py develop

# Install additional requirements
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
```

## Usage

To evaluate trackers on the EMT dataset:
```bash
pip install -r requirements.txt
```
```bash
python evaluate_trackers.py
```

## Project Structure

```
.
├── evaluate_trackers.py            # Main evaluation script
├── test_trackers.py                # Main tracker test script
├── emt/              
│   └── emt_annotations             # EMT annotaion labels and seqmal
|   |   └──labels
|   |   └──evaluate_tracking.seqmap  # To be used by trackeval
|   └── frames
|   └── raw
└── requirements.txt       # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Add your license information here]