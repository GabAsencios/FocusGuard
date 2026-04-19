# FocusGuard

A real-time distraction detection system that monitors both webcam and screen activity to help users maintain focus. FocusGuard uses YOLOv8 for webcam-based object detection (phone usage and user absence) and a fine-tuned ResNet18 model for screen content classification.

## Features

- **Webcam Monitoring**: Detects phone usage and user absence using YOLOv8
- **Screen Classification**: Classifies screen content as Gaming, Productive, Twitch, or YouTube
- **Real-time Alerts**: Audio alerts for detected distractions
- **Temporal Filtering**: Reduces false positives with grace periods and majority voting
- **GPU Acceleration**: CUDA support for faster inference

Custom Datasets: https://huggingface.co/datasets/GabAsencios/FocusGuard/tree/main/data

Trained Models: https://huggingface.co/datasets/GabAsencios/FocusGuard/tree/main/models

## Directory Structure

```
FocusGuard/
├── data/                           # Training data
│   ├── Gaming/
│   ├── Productive/
│   ├── Twitch/
│   └── YouTube/
├── models/                         # Pre-trained model weights
│   ├── resnet18_screen_AdamW_model.pth
│   └── yolov8m.pt
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── FocusGuard_CamDetection.ipynb
│   ├── FocusGuard_ScreenClassifier.ipynb
│   └── FocusGuard_Testings.ipynb
├── src/                           # Source code
│   ├── __init__.py
│   ├── webcam_module.py          # Webcam detection module
│   ├── screen_classifier.py      # Screen classification module
│   └── datasets/                 # Dataset utilities
├── TestResults/                   # Hyperparameter testing results
│   ├── adamw_results.json
│   ├── hyperparameter_results.json
│   └── hyperparameterTestingResults.txt
├── DatasetBuilder.py              # Dataset creation utility
├── main.py                        # Main application entry point
└── README.md
```

## Dependencies

### Core Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (with CUDA support recommended)
- **OpenCV**: 4.x
- **Ultralytics**: Latest (for YOLOv8)
- **torchvision**: Latest
- **Pillow**: Latest
- **mss**: Latest (for screen capture)
- **numpy**: Latest

### Installation

1. **Clone the repository**:
   ```bash
   git clone [<repository-url>](https://github.com/GabAsencios/FocusGuard)
   cd FocusGuard
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install ultralytics opencv-python pillow mss numpy
   ```

   > **Note**: Adjust the CUDA version in the PyTorch installation command based on your GPU. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the correct command.

4. **Verify GPU support** (optional but recommended):
   ```python
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
   ```

## Model Weights

Before running FocusGuard, ensure you have the required model weights in the `models/` directory:

1. **YOLOv8 Model** (`yolov8m.pt`):
   - Download from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
   - Place in `models/yolov8m.pt`

2. **ResNet18 Screen Classifier** (`resnet18_screen_AdamW_model.pth`):
   - Custom trained model (should already be included in the repository)
   - Path: `models/resnet18_screen_AdamW_model.pth`

## Running FocusGuard

### Quick Start

Run the main application:

```bash
python main.py
```

### Configuration

You can modify the following parameters in `main.py`:

- **`CAPTURE_INTERVAL`**: Screenshot capture interval in seconds (default: 3)

#### Webcam Module Parameters (in `src/webcam_module.py`):

- `model_path`: Path to YOLOv8 weights
- `conf_threshold`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `event_threshold`: Seconds a distraction must persist to trigger (default: 3)
- `grace_period`: Seconds to wait before resetting timer (default: 0.5)

#### Screen Classifier Parameters (in `src/screen_classifier.py`):

- `model_path`: Path to ResNet18 weights
- `num_classes`: Number of screen classes (default: 4)
- `conf_threshold`: Minimum confidence to trigger alert (default: 0.6)
- `monitor_index`: Monitor to capture (1 = primary, 2 = secondary)
- `dropout_rate`: Dropout rate matching training config (default: 0.5)


## Controls

- **Press `q`**: Quit the application
- The webcam feed window displays real-time detection results with bounding boxes

## Alerts

### Webcam Alerts

- **Phone Distraction**: High-pitched beep (1000 Hz, 300ms) when phone usage is detected for 3+ seconds
- **User Absent**: Low-pitched beep (500 Hz, 300ms) when user is absent for 3+ seconds

### Screen Alerts

- **Distractor Detected**: Medium beep (1000 Hz, 500ms) when Gaming, Twitch, or YouTube is detected with confidence ≥ 0.6

## Troubleshooting

### Common Issues

1. **Webcam not detected**:
   - Ensure your webcam is connected and not being used by another application
   - Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

2. **CUDA not available**:
   - Verify your GPU drivers are up to date
   - Reinstall PyTorch with the correct CUDA version
   - The application will fall back to CPU if CUDA is unavailable

3. **Model files not found**:
   - Ensure model weights are placed in the `models/` directory
   - Check file paths in `main.py` match your actual file locations

4. **Screen capture issues**:
   - Adjust `MONITOR_INDEX` in `src/screen_classifier.py` for multi-monitor setups
   - Ensure `mss` library has necessary permissions on your system

5. **Audio alerts not working** (Windows):
   - `winsound` is Windows-only. For Linux/Mac, replace with alternative audio libraries

## System Requirements

### Minimum

- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 4GB
- **Camera**: Any USB/built-in webcam
- **Python**: 3.8+

### Recommended

- **OS**: Windows 10/11
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 2070 or better)
- **Camera**: 720p or higher resolution webcam
- **Python**: 3.9+

## Performance

- **Webcam FPS**: ~30 FPS on GPU, ~10-15 FPS on CPU
- **Screen Classification**: Every 3 seconds (configurable)
- **Memory Usage**: ~2-4GB with GPU, ~1-2GB with CPU

## Contributing

Contributions are welcome! Please ensure code follows the existing structure and includes appropriate documentation.

## License



## Authors

Gabriel Asencios

## Acknowledgments

- YOLOv8 by Ultralytics
- ResNet18 architecture by PyTorch/torchvision
- COCO dataset for object detection classes
