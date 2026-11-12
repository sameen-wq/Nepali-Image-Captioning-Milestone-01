"""Minimal, reproducible ResNet18 feature-extraction demonstration.

From repository root:
  python -m src.run_experiment --weights pretrained
  python -m src.run_experiment --weights random  # code-only smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps

from .encoder import WEIGHT_ID, build_encoder, encode

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def printable_path(path: Path) -> str:
    """Use a portable repo-relative path where possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def experiment(first: Path, second: Path, weight_mode: str, output_dir: Path, device: str = "cpu") -> dict:
    for p in (first, second):
        if not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model, preprocess = build_encoder(weight_mode, device)

    with Image.open(first) as im:
        image_a = im.convert("RGB")
    with Image.open(second) as im:
        image_b = im.convert("RGB")

    first_vector = encode(image_a, model, preprocess, device)
    repeat_vector = encode(image_a, model, preprocess, device)
    flipped_vector = encode(ImageOps.mirror(image_a), model, preprocess, device)
    other_vector = encode(image_b, model, preprocess, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors = torch.stack([first_vector, repeat_vector, flipped_vector, other_vector]).numpy()
    np.savez_compressed(output_dir / "embeddings.npz", embeddings=vectors, labels=np.array(
        ["image_a", "image_a_repeat", "image_a_flipped", "image_b"]
    ))
    metrics = {
        "experiment": "Milestone 01: ResNet18 image feature extraction",
        "date": str(date.today()),
        "status": "SMOKE TEST ONLY - RANDOM MODEL; NOT SCIENTIFIC EVIDENCE" if weight_mode == "random" else "PRETRAINED ENCODER DEMONSTRATION; NOT A CAPTION-QUALITY EVALUATION",
        "weights": "random initialization (seed=42)" if weight_mode == "random" else f"TorchVision ResNet18_Weights.{WEIGHT_ID} (ImageNet-1K)",
        "backbone": "resnet18, final fully connected classification head replaced with Identity",
        "embedding_shape_single": list(first_vector.shape),
        "preprocessing": "Official ResNet18 ImageNet-1K V1 weights transforms: resize 256, center crop 224, normalize",
        "images": [
            {"path": printable_path(first), "sha256": sha256(first)},
            {"path": printable_path(second), "sha256": sha256(second)},
        ],
        "comparisons": {
            "same_image_repeated_cosine": cosine(first_vector, repeat_vector),
            "horizontally_flipped_cosine": cosine(first_vector, flipped_vector),
            "different_image_cosine": cosine(first_vector, other_vector),
        },
        "first_image_embedding_sample_first_8": [float(v) for v in first_vector[:8]],
        "first_image_embedding_l2_norm": float(torch.linalg.vector_norm(first_vector)),
        "deterministic_repeat_exact_equal": bool(torch.equal(first_vector, repeat_vector)),
        "environment": {
            "python_torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "random_seed": 42,
        },
        "interpretation": "Embeddings are computational features, NOT Nepali captions or a test of caption quality. Cosine similarities are descriptive for these specific example images only.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "embedding_shape": metrics["embedding_shape_single"], "comparisons": metrics["comparisons"], "saved_to": str(output_dir)}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Reproducible ResNet18 feature extraction demo")
    parser.add_argument("--weights", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--image-a", type=Path, default=ROOT / "assets/astronaut.png")
    parser.add_argument("--image-b", type=Path, default=ROOT / "assets/chelsea.png")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu")
    out = args.output_dir or ROOT / "results" / ("pretrained_resnet18" if args.weights == "pretrained" else "smoke_test_random_weights")
    experiment(args.image_a, args.image_b, args.weights, out, args.device)


if __name__ == "__main__":
    main()
