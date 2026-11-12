# Nepali Image Caption Generation Using Reformer

**Status:** Ongoing faculty-supervised research — **Milestone 01: ResNet18 image feature extraction**

This repository is a small, independently reproducible **visual-encoder experiment**, not a completed Nepali captioning system. The broader study aims to create a model that generates meaningful Nepali-language image captions, improve or assemble a Nepali image–caption dataset, and evaluate a Reformer-based approach against credible baselines.

> **Scope and scientific honesty:** No Nepali caption generator, Reformer, supervised caption training, caption-quality evaluation, or speed/memory advantage is implemented or claimed in this milestone. Reformer efficiency on long sequences does not itself demonstrate lower data requirements or superiority on short Nepali captions.

## Why this milestone?

Captioning requires visual representations before language generation. We first test whether a standard pretrained CNN yields fixed-size features that can eventually be projected or otherwise supplied to a caption decoder. This is a **pipeline and interface check**, not evidence the encoder understands Nepali or produces captions.

## Architecture (proposed)

```text
Image -> pretrained ResNet18 -> 512-D image embedding -> [future: projection / image conditioning]
                                                             |
                                              [future: Reformer language model]
                                                             |
                                              [future: Nepali caption]
```

The visual-to-language interface and whether a Reformer decoder is suitable remain open design questions.

## What is implemented?

- Official TorchVision ImageNet-1K V1 preprocessing: resize, center crop, normalization.
- Frozen ResNet18 without its classification head, giving **512-dimensional** image features.
- Repeatability check for one image, a horizontal-mirror comparison, and a different-image comparison.
- Reproducibility metadata: weight mode, file SHA-256, PyTorch/TorchVision versions and saved vectors.
- A pytest smoke test and an exploratory notebook.

## Sample images

| File | Description | Reuse provenance |
| --- | --- | --- |
| `assets/astronaut.png` | Eileen Collins astronaut photograph | NASA image distributed by `skimage.data.astronaut()`; public domain per scikit-image docs. [Source](https://scikit-image.org/docs/stable/api/skimage.data) |
| `assets/chelsea.png` | Chelsea the cat | Photo by Stefan van der Walt, CC0; distributed by `skimage.data.chelsea()`. [Source](https://scikit-image.org/docs/stable/api/skimage.data) |

These two photos are **not** a Nepali caption dataset. No labels or captions are used.

## How to run

**Requirements:** Python 3.10+ recommended, working Torch/TorchVision installation, and internet on first pretrained run (~45 MB checkpoint). From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Real experiment: downloads torchvision ImageNet-1K pretrained weights on first use
python -m src.run_experiment --weights pretrained

# Reproducibility/software test only (random features: scientifically meaningless)
python -m src.run_experiment --weights random

# Automated checks
python -m pytest -q
```

Use `--image-a path/to/photo1.jpg --image-b path/to/photo2.jpg` to substitute images you have permission to use. Avoid publishing private or restricted dataset images.

## Outputs

- `results/pretrained_resnet18/metrics.json` from the **pretrained** run; generated only after it actually succeeds.
- `results/pretrained_resnet18/embeddings.npz` with four 512-D vectors: original A, repeated A, mirrored A, and B.
- `results/smoke_test_random_weights/metrics.json` is packaged with the example **random-weights smoke test** to demonstrate code execution. **This is not a pretrained scientific result.**

The JSON reports exact embedding shape, representative numeric values, cosine comparisons, SHA-256 image hashes, seed and environment versions. The exact numeric features may vary by framework version, runtime and processing hardware.

## Questions and limitations

1. A 512-D vector is a global feature summary. Its existence alone does not prove semantic fidelity or object-level recognition.
2. One flipped photo and one different photo do not establish invariance or embedding quality; these comparisons only validate the experiment interface.
3. Pretrained ImageNet encoders may miss Nepal-specific objects or cultural context. A proper study needs an approved image dataset, held-out evaluation and visual backbones compared under matched settings.
4. The Reformer is designed for memory efficiency (particularly on long sequences), but *caption length, dataset size, conditioning method and memory/accuracy tradeoffs must be empirically tested*. Avoid claiming it is automatically more data-efficient.

## Next research milestones (not implemented)

1. Run and report the pretrained ResNet18 experiment, and evaluate on authorized target-domain images.
2. Compare ResNet18 with another backbone using identical preprocessing and sample splits, reporting compute and visual-feature quality appropriately.
3. Design an image-conditioning mechanism for a Reformer LM (for example, a visual prefix) and compare it with a standard Transformer captioning baseline.
4. Document Nepali image-caption dataset provenance, linguistic variation, annotations and train/validation/test splits. Later evaluate captions with automatic metrics and Nepali-fluent human review.

## Research provenance & sharing

This is a **demonstration scaffold created for the ongoing project**, not a claim that an existing laboratory experiment or professor-approved dataset has already been run. Add your own verified experiments and contributions. Obtain your professor's approval before publicly posting jointly developed methods, unpublished findings, code or data. Review `RESEARCH_LOG.md` and `results/README.md` before describing milestones on a CV.

## References

- [PyTorch TorchVision ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18)
- [TorchVision pretrained model/transforms usage](https://docs.pytorch.org/vision/stable/models.html)
- [Kitaev, Kaiser, and Levskaya, *Reformer: The Efficient Transformer* (ICLR 2020)](https://arxiv.org/abs/2001.04451)

See [`GITHUB_UPLOAD_GUIDE.md`](GITHUB_UPLOAD_GUIDE.md) for a step-by-step publishing checklist.
