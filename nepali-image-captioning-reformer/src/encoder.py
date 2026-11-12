"""ResNet18 penultimate-layer image representations (not captions)."""
from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

WEIGHT_ID = "IMAGENET1K_V1"


def build_encoder(weight_mode: str = "pretrained", device: str = "cpu") -> tuple[nn.Module, object]:
    """Return a frozen ResNet18 and the *pretrained weights' official* transforms.

    `random` is a smoke-test mode only; it has no semantic validity.
    TorchVision downloads pretrained weights at first use (~45 MB).
    """
    if weight_mode not in {"pretrained", "random"}:
        raise ValueError("weight_mode must be 'pretrained' or 'random'")
    weights = ResNet18_Weights.IMAGENET1K_V1 if weight_mode == "pretrained" else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if weight_mode == "pretrained":
            raise RuntimeError(
                "Could not load ResNet18 pretrained weights. The first run requires "
                "internet access to download TorchVision ImageNet-1K weights. "
                "Do NOT substitute random weights for research findings; "
                "use --weights random only as a clearly labeled smoke test."
            ) from exc
        raise
    model.fc = nn.Identity()  # 512 features instead of 1,000 class logits.
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, preprocess


@torch.inference_mode()
def encode(image: Image.Image, model: nn.Module, preprocess: object, device: str = "cpu") -> torch.Tensor:
    """One RGB image -> a CPU tensor of shape (512,); no gradient computation."""
    image_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    embedding = model(image_tensor)
    if embedding.shape != (1, 512):
        raise AssertionError(f"Unexpected ResNet18 embedding dimensions: {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise AssertionError("Non-finite embedding values detected")
    return embedding.squeeze(0).cpu()
