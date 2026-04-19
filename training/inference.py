"""
Inference script for CS2 navigation model.
Loads trained model and predicts actions from images in real-time or batch mode.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image

from training.model import create_model
from training.label_maps import LabelEncoder


class NavigationPredictor:
    """Wrapper for model inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        history_len: int = 1,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.history_len = history_len

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})

        # Create model
        self.model = create_model(
            backbone=config.get("backbone", "resnet18"),
            n_move_classes=9,
            n_turn_classes=9,
            history_len=history_len,
            use_temporal=config.get("use_temporal", False),
            pretrained=False,
        )

        # Load weights (handle DDP wrapper)
        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Label encoder
        self.label_encoder = LabelEncoder()

        # Frame buffer for history
        self.frame_buffer = []

        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image) -> dict:
        """
        Predict actions from a single image.

        Args:
            image: RGB image as numpy array (H, W, 3) or PIL Image

        Returns:
            Dict with predicted actions
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Resize to 224x224
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0

        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # Add to buffer
        self.frame_buffer.append(image_tensor)
        if len(self.frame_buffer) > self.history_len:
            self.frame_buffer.pop(0)

        # Pad buffer if needed
        while len(self.frame_buffer) < self.history_len:
            self.frame_buffer.insert(0, self.frame_buffer[0])

        # Stack to (1, T, C, H, W)
        batch = torch.stack(self.frame_buffer, dim=0).unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(batch)

        # Decode predictions
        move_idx = outputs["move"].argmax(dim=1).item()
        turn_idx = outputs["turn"].argmax(dim=1).item()
        jump = torch.sigmoid(outputs["jump"]).item() > 0.5
        crouch = torch.sigmoid(outputs["crouch"]).item() > 0.5
        fire = torch.sigmoid(outputs["fire"]).item() > 0.5
        mouse_dx = outputs["mouse_dx"].item()
        mouse_dy = outputs["mouse_dy"].item()

        return {
            "action_move": self.label_encoder.decode_move(move_idx),
            "action_turn": self.label_encoder.decode_turn(turn_idx),
            "action_jump": jump,
            "action_crouch": crouch,
            "action_fire": fire,
            "mouse_dx": mouse_dx,
            "mouse_dy": mouse_dy,
            "move_confidence": torch.softmax(outputs["move"], dim=1)[0, move_idx].item(),
            "turn_confidence": torch.softmax(outputs["turn"], dim=1)[0, turn_idx].item(),
        }

    def reset_history(self) -> None:
        """Clear frame buffer."""
        self.frame_buffer = []


def demo_inference(checkpoint_path: str, image_path: str) -> None:
    """Demo inference on a single image."""
    predictor = NavigationPredictor(checkpoint_path, device="cuda", history_len=1)

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Predict
    start = time.time()
    prediction = predictor.predict(image)
    elapsed = time.time() - start

    print(f"\nPrediction (took {elapsed*1000:.1f}ms):")
    print(json.dumps(prediction, indent=2))


def batch_inference(checkpoint_path: str, image_dir: str, output_path: str) -> None:
    """Run inference on a directory of images."""
    predictor = NavigationPredictor(checkpoint_path, device="cuda", history_len=1)

    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))

    results = []
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        prediction = predictor.predict(image)
        prediction["image_path"] = str(img_path)
        results.append(prediction)

        print(f"Processed {img_path.name}: {prediction['action_move']}, {prediction['action_turn']}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CS2 navigation model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--image", type=str, help="Single image path for demo")
    parser.add_argument("--image-dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON path")
    parser.add_argument("--history-len", type=int, default=1, help="History length")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    if args.image:
        demo_inference(args.checkpoint, args.image)
    elif args.image_dir:
        batch_inference(args.checkpoint, args.image_dir, args.output)
    else:
        print("Please specify --image or --image-dir")


if __name__ == "__main__":
    main()
