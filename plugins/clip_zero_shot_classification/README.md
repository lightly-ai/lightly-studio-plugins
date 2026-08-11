# CLIP Zero-Shot Classification Plugin

Zero-shot image classification using [CLIP](https://huggingface.co/openai/clip-vit-base-patch16). You define the class vocabulary as a table of text prompts and the label each prompt should assign — no training and no fixed label set. Runs on image collections in Lightly Studio.

Each image is scored against every prompt, and the label of the best-matching prompt is written as a classification annotation.

## Setup

### 1. Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/clip_zero_shot_classification/"
```

### 2. GPU (optional)

The plugin picks a device automatically: CUDA, else MPS, else CPU. To use a CUDA GPU, reinstall PyTorch with the appropriate CUDA build:

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
| `prompt` | yes | CLIP text prompt, e.g. `"a photo of a dog"`. A row with an empty prompt is an error, not a skipped row |
| `label` | no | Label to assign when this prompt scores highest. Defaults to the prompt itself |

Add and remove rows in the GUI to define your classes. Prompt wording matters for CLIP — `"a photo of a dog"` typically works better than a bare `"dog"`.

Rows sharing a label merge into a single annotation class, so `a photo of a wolf` and `a photo of a dog` can both map to `canine`.

## Notes

Scores are a softmax over the prompts you supply, so they say which prompt fits best, not whether any of them fits at all.

- An image whose true class is not in the table is still labelled with the closest prompt.
- The `confidence_threshold` only catches such images while every prompt is equally wrong. Given `dog` and `cat`, a photo of a train splits about 0.58/0.42 and a threshold of 0.7 discards it; with a prompt it fits better, like `red vehicle`, the score saturates above 0.99 and no threshold filters it.
- Scores are relative to the whole list, so adding or removing one prompt shifts all the others.

Each image gets at most one `classification` annotation. Unreadable images are skipped and counted in the result message.
