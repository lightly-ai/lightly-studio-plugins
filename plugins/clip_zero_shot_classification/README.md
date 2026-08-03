# CLIP Zero-Shot Classification Plugin

Zero-shot image classification using [CLIP](https://huggingface.co/openai/clip-vit-base-patch16). You define the class vocabulary as a table of text prompts and the label each prompt should assign — no training and no fixed label set. Runs on image collections in Lightly Studio.

Each image is scored against every prompt, and the label of the best-matching prompt is written as a classification annotation.

## Setup

### 1. Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/clip_zero_shot_classification/"
```

### 2. GPU (optional)

By default the plugin runs on CUDA if available, then Apple Silicon (MPS), then CPU. To use a CUDA GPU, reinstall PyTorch with the appropriate CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_id` | string | `"openai/clip-vit-base-patch16"` | HuggingFace CLIP model ID — see below |
| `prompts` | table | dog / cat example rows | Table of `prompt` and `label` pairs — see below |
| `confidence_threshold` | float | `0.0` | Minimum score for assigning a label. Images below this are left unclassified |
| `annotation_source` | string | `"clip_zero_shot"` | Target annotation source where predictions are stored. Override this to store results separately from an earlier run |

### Choosing a model

Any CLIP checkpoint on HuggingFace that loads with `CLIPModel` works. Larger models are more accurate but download more weights and run slower:

| Model | Notes |
|---|---|
| `openai/clip-vit-base-patch32` | Fastest, smallest download |
| `openai/clip-vit-base-patch16` | **Default.** Same size class as patch32 but finer patches, noticeably more accurate |
| `openai/clip-vit-large-patch14` | Most accurate of the OpenAI checkpoints, ~3x the weights |
| `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | LAION-trained, strongest zero-shot accuracy, slowest |

### The `prompts` table

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | CLIP text prompt, e.g. `"a photo of a dog"` |
| `label` | no | Label to assign when this prompt scores highest. Defaults to the prompt itself |

Add and remove rows in the GUI to define your classes. Prompt wording matters for CLIP — `"a photo of a dog"` typically works better than a bare `"dog"`.

**Several rows may share a label** to ensemble multiple prompts for one class. The scores of rows sharing a label are pooled with a max, so extra phrasings strengthen the class instead of splitting it:

| prompt | label |
|---|---|
| `a photo of a dog` | `dog` |
| `a dog in a park` | `dog` |
| `a photo of a cat` | `cat` |

## Notes

- **Cover your data with the vocabulary.** By far the most common cause of bad results is a
  missing class rather than a weak model: because the scores are a softmax over the prompts
  you supply, an image whose true class is absent is still forced onto the closest prompt.
  If you only define `dog` and `cat`, a photo of a train confidently becomes one of them.
  Add a row for every class you expect to see.
- Each image receives at most one `classification` annotation — the single best-matching label.
- Scores are a softmax across all prompts, so they are relative to the vocabulary you define. Adding or removing prompts changes the scores of the others, and a `confidence_threshold` tuned for one vocabulary will not transfer to another.
- Because the softmax is over the prompts you supply, an image that matches none of them still gets a high score for the closest one. Add explicit "background" or "other" rows if you want a catch-all class rather than relying on the threshold.
- Images that cannot be read are skipped and reported in the result message.
