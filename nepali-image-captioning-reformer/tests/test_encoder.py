"""Unit tests rely on random weights, and do NOT validate pretrained performance."""
import torch
from PIL import Image
from src.encoder import build_encoder, encode


def test_feature_shape_and_repeatability():
    torch.manual_seed(42)
    model, preprocess = build_encoder("random")
    img = Image.new("RGB", (320, 320), color=(128, 90, 60))
    a = encode(img, model, preprocess)
    b = encode(img, model, preprocess)
    assert a.shape == (512,)
    assert torch.isfinite(a).all()
    assert torch.equal(a, b)
    assert all(not p.requires_grad for p in model.parameters())
