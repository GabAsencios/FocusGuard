import torch
import mss
import numpy as np
from PIL import Image
from torchvision import models, transforms
import winsound
from collections import deque

# ---- Configuration ----

SCREEN_CLASSES     = ['Gaming', 'Productive', 'Twitch', 'YouTube']  # must match ImageFolder alphabetical order
DISTRACTOR_CLASSES = ['Gaming', 'Twitch', 'YouTube']
CONF_THRESHOLD     = 0.6    # minimum confidence to trigger a distraction alert
MONITOR_INDEX      = 1      # change for monito (1 = primary, 2 = secondary)
DROPOUT_RATE       = 0.5    # must match training config
NUM_CLASSES        = 4      # must match training config
MAJORITY_VOTE_N    = 3      # number of consecutive predictions to majority vote over


class ScreenClassifier:
    """
    Screen-based distraction classifier using a fine-tuned ResNet18 model.

    Captures a screenshot from the specified monitor, classifies it into
    one of four classes (Gaming, Productive, Twitch, YouTube), and prints
    an alert if a distractor class is detected above the confidence threshold.
    Uses a majority vote buffer to filter single-frame misclassifications.

    Args:
        model_path (str): Path to the saved ResNet18 .pth weights file.
        num_classes (int): Number of output classes (default: 4).
        conf_threshold (float): Minimum confidence to trigger alert (default: 0.6).
        monitor_index (int): Monitor to capture (1 = primary, 2 = secondary).
        dropout_rate (float): Dropout rate matching training config (default: 0.5).

    Example:
        classifier = ScreenClassifier(model_path='models/resnet18_screenshot_ADAM_model.pth')
        classifier.detect_and_alert()
    """

    def __init__(self,
                 model_path,
                 num_classes=NUM_CLASSES,
                 conf_threshold=CONF_THRESHOLD,
                 monitor_index=MONITOR_INDEX,
                 dropout_rate=DROPOUT_RATE):

        self.conf_threshold     = conf_threshold
        self.monitor_index      = monitor_index
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model              = self._load_model(model_path, num_classes, dropout_rate)
        self.transform          = self._build_transform()
        self.prediction_buffer  = deque(maxlen=MAJORITY_VOTE_N)
        print(f'ScreenClassifier loaded on {self.device}')

    def _load_model(self, model_path, num_classes, dropout_rate):
        """
        Load fine-tuned ResNet18 weights from disk.

        Rebuilds the exact FC architecture used during training:
        Sequential(Dropout, Linear) to match the saved state dict.

        Args:
            model_path (str): Path to the .pth weights file.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate used during training.

        Returns:
            torch.nn.Module: Loaded ResNet18 model in eval mode.

        Example:
            model = self._load_model('models/resnet18_screenshot_ADAM_model.pth', 4, 0.5)
        """
        model = models.resnet18(weights=None)

        # Save in_features before replacing fc
        in_features = model.fc.in_features

        # Rebuild exact FC architecture from training — must match saved weights
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(in_features, num_classes)
        )

        # Load saved weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set to eval mode to disable dropout during inference
        model.eval()
        model.to(self.device)
        return model

    def _build_transform(self):
        """
        Build the preprocessing pipeline matching val/test transforms used during training.

        Applies resize to 224x224 and ImageNet normalization — no augmentation.

        Returns:
            torchvision.transforms.Compose: Preprocessing pipeline.

        Example:
            transform = self._build_transform()
            tensor = transform(image).unsqueeze(0)
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def capture_screen(self):
        """
        Capture a screenshot from the configured monitor.

        Args:
            None

        Returns:
            PIL.Image: Captured screenshot as an RGB PIL image.

        Example:
            img = self.capture_screen()
        """
        with mss.mss() as sct:
            img = sct.grab(sct.monitors[self.monitor_index])
            return Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')

    def classify(self, image):
        """
        Run inference on a PIL image and return predicted class and confidence.

        Args:
            image (PIL.Image): Screenshot to classify.

        Returns:
            tuple: (class_name (str), confidence (float))

        Example:
            label, conf = self.classify(img)
            # Returns: ('YouTube', 0.94)
        """
        # Preprocess image into a batch tensor
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            # Convert logits to probabilities
            probs      = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

        class_name = SCREEN_CLASSES[pred.item()]
        confidence = conf.item()
        return class_name, confidence

    def detect_and_alert(self):
        """
        Capture the screen, classify it, and print an alert and play a sound
        if a distractor is detected above the confidence threshold.

        Uses a majority vote buffer of the last 3 predictions to filter
        single-frame misclassifications before triggering an alert.

        Args:
            None

        Returns:
            tuple: (class_name (str), confidence (float))

        Example:
            classifier.detect_and_alert()
            # [SCREEN ALERT] Distraction detected: YouTube (confidence: 0.94)
        """
        # Capture and classify the current screen
        screenshot             = self.capture_screen()
        class_name, confidence = self.classify(screenshot)

        # Add prediction to majority vote buffer
        self.prediction_buffer.append(class_name)

        # Only alert if majority of last N predictions agree
        majority_class = max(set(self.prediction_buffer),
                             key=self.prediction_buffer.count)
        majority_count = self.prediction_buffer.count(majority_class)

        # Require at least 2 out of 3 predictions to agree before alerting
        if (majority_class in DISTRACTOR_CLASSES and
                confidence >= self.conf_threshold and
                majority_count >= 2):
            print(f'[SCREEN ALERT] Distraction detected: {majority_class} '
                  f'(confidence: {confidence:.2f})')
            # Play beep: frequency 1000Hz, duration 500ms
            winsound.Beep(1000, 500)
        else:
            print(f'[SCREEN] Productive ({class_name}: {confidence:.2f})')

        return class_name, confidence