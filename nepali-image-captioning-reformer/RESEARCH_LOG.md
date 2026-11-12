# Research log — Milestone 01

**Research objective (supervisor's description, summarized):** Explore Nepali image captioning with image understanding and a Reformer-based language component; develop or enhance Nepali image-caption data; and compare with existing models. Claims about lower compute cost and improved caption quality are **hypotheses to evaluate**, not established findings for this study.

## Milestone 01 — Encoder experiment

**Why this experiment?** Before connecting image features to a language model, confirm the visual encoder returns stable, fixed-dimensional numerical features from diverse images. This establishes a minimal, independently testable interface for later visual-to-language integration.

**Questions**
1. Does the ResNet18 feature extractor return a valid 512-dimensional embedding for each image?
2. Is repeated inference on the same image deterministic in evaluation mode?
3. How do cosine similarities change for a mirrored image versus a second, visibly different image? (Exploratory only.)

**Method**: TorchVision ResNet18, remove 1000-class classifier by replacing `fc` with `Identity`, use official pretrained ImageNet-1K preprocessing and weights for actual research demonstration. Compute embeddings for two open sample photos and a horizontal mirror. Save cosine values, hashes, software versions and 512-D vectors.

**Implemented and locally validated (2026-11-12):** The experiment script and a randomized-weight *software smoke test* (not pretrained evidence).

**Not yet validated in this package:** Pretrained weight download and numerical pretrained results; Nepali text captions; Reformer model; dataset creation; training; benchmarks or compute/memory claims.

**Important limitations**: ImageNet features may not preserve information relevant to Nepali descriptions or local cultural objects. Global average pooled embeddings discard detailed spatial structure. Two sample photos cannot establish task performance or robustness. Cosine similarity is not a quality metric for generated captions. The pretrained ImageNet backbone must be tested before concluding it is appropriate for the target dataset.

**Next steps**: Run `--weights pretrained` and record verified outputs; evaluate sample images from the approved Nepali caption dataset; compare against ResNet50 or other visual backbones under consistent conditions; explore spatial features or visual prefix projection and compare decoder/baseline architectures in a separate milestone.
