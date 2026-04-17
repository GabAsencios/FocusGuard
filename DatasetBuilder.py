"""
FocusGuard - Screenshot Dataset Capture Script

Captures screenshots at regular intervals and saves them with class labels.

Usage:
    python DatasetBuilder.py --class_name Gaming --output_dir dataset --interval 3

Arguments:
    --class_name  : One of [YouTube, Twitch]
    --output_dir  : Root folder where screenshots will be saved (default: dataset)
    --interval    : Seconds between captures (default: 3)

Example:
    python capture_screenshots.py --class_name YouTube --output_dir dataset --interval 3
"""

import argparse
import os
import time
import mss
from datetime import datetime

from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_CLASSES = ["YouTube", "Twitch","Gaming"]
DEFAULT_INTERVAL = 3
DEFAULT_OUTPUT_DIR = "dataset"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with fields:
            - class_name (str)
            - output_dir (str)
            - interval (int)

    Example:
        args = parse_args()
        print(args.class_name)  # "Gaming"
    """
    parser = argparse.ArgumentParser(description="FocusGuard screenshot capture tool")
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        choices=VALID_CLASSES,
        help=f"Class label for this session. One of: {VALID_CLASSES}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory to save screenshots (default: dataset)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Seconds between screenshots (default: 3)",
    )
    return parser.parse_args()


def setup_output_dir(output_dir: str, class_name: str) -> str:
    """
    Create the output directory for the given class if it doesn't exist.

    Args:
        output_dir (str): Root dataset directory.
        class_name (str): Class label (subfolder name).

    Returns:
        str: Full path to the class subfolder.

    Example:
        path = setup_output_dir("dataset", "Gaming")
        # Creates ./dataset/Gaming/ and returns its path
    """
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    return class_dir


def capture_screenshot(class_dir: str, class_name: str, index: int) -> str:
    """
    Capture a full-resolution screenshot and save it as a PNG.

    Filename format: {class_name}_{timestamp}_{index:05d}.png

    Args:
        class_dir (str): Directory to save the screenshot.
        class_name (str): Class label used in the filename.
        index (int): Sequential capture index for ordering.

    Returns:
        str: Full path to the saved screenshot file.

    Example:
        path = capture_screenshot("dataset/Gaming", "Gaming", 42)
        # Saves dataset/Gaming/Gaming_20260416_143022_00042.png
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{class_name}_{timestamp}_{index:05d}.png"
    filepath = os.path.join(class_dir, filename)

    with mss.mss() as sct:
        img = sct.grab(sct.monitors[2])
        screenshot = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
    screenshot.save(filepath)

    return filepath


def run_capture_session(class_name: str, output_dir: str, interval: int) -> None:
    """
    Run an interactive screenshot capture session until the user stops it.

    Prints progress to stdout. Press Ctrl+C to stop.

    Args:
        class_name (str): Class label for this session.
        output_dir (str): Root directory for saving screenshots.
        interval (int): Seconds to wait between captures.

    Returns:
        None

    Example:
        run_capture_session("YouTube", "dataset", 3)
    """
    class_dir = setup_output_dir(output_dir, class_name)

    print(f"\nFocusGuard Dataset Capture")
    print(f"  Class     : {class_name}")
    print(f"  Output    : {class_dir}")
    print(f"  Interval  : {interval}s")
    print(f"\nSwitch to the window you want to capture, then press Enter to start...")
    input()

    print("Capturing... Press Ctrl+C to stop.\n")

    index = 0
    try:
        while True:
            filepath = capture_screenshot(class_dir, class_name, index)
            index += 1
            print(f"[{index:>4}] Saved: {os.path.basename(filepath)}")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped. {index} screenshots saved to: {class_dir}")
        print_session_summary(class_dir, class_name, index)


def print_session_summary(class_dir: str, class_name: str, count: int) -> None:
    """
    Print a summary of the capture session including file count and folder size.

    Args:
        class_dir (str): Directory where screenshots were saved.
        class_name (str): Class label for this session.
        count (int): Number of screenshots captured this session.

    Returns:
        None

    Example:
        print_session_summary("dataset/Gaming", "Gaming", 120)
    """
    total_files = len([f for f in os.listdir(class_dir) if f.endswith(".png")])
    total_size_mb = sum(
        os.path.getsize(os.path.join(class_dir, f))
        for f in os.listdir(class_dir)
        if f.endswith(".png")
    ) / (1024 * 1024)

    print(f"\n--- Session Summary ---")
    print(f"  Class           : {class_name}")
    print(f"  This session    : {count} screenshots")
    print(f"  Total in folder : {total_files} screenshots")
    print(f"  Folder size     : {total_size_mb:.1f} MB")
    print(f"-----------------------")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_capture_session(
        class_name=args.class_name,
        output_dir=args.output_dir,
        interval=args.interval,
    )