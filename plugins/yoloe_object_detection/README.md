# YOLOE Open Vocabulary Object Detection Plugin

Auto-labeling with [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/) models. Runs YOLOE inference on a table of text prompts of your own choosing — no retraining needed — and adds bounding box or instance segmentation annotations to images in Lightly Studio.

## Setup

### Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yoloe_object_detection/"
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `"yoloe-26n-seg.pt"` | YOLOE model weights path or Ultralytics model name (e.g. `yoloe-11s-seg.pt`, `yoloe-v8l-seg.pt`, `/path/to/custom.pt`) |
| `confidence` | float | `0.25` | Minimum confidence threshold for keeping a prediction |
| `prompts` | table | one row, `person` | Prompts to detect with and the labels to assign. See below |
| `instance_segmentation` | bool | `false` | Store instance segmentation masks instead of bounding boxes |
| `annotation_source` | string | `"yoloe_auto_label__{model_path}"` | Target annotation source name where predictions will be stored |

### The `prompts` table

| Column | Required | Description |
|---|---|---|
| `prompt` | yes | What to detect, e.g. `"person"` |
| `label` | no | Annotation label for this prompt's detections. Leave empty to use the prompt |

Add rows to detect several concepts in one run. Rows sharing a label merge into a single
annotation class, so `car` and `truck` can both map to `vehicle`. Empty rows and repeated
prompts are rejected — remove them rather than leaving them blank.

## Notes

- Unlike the [YOLO plugin](../yolo_object_detection/), classes are not fixed by the checkpoint — detect anything you name in the `prompts` table, including multi-word phrases like `stuffed toy`. Missing labels are created in the dataset.
- Prompt-capable checkpoints are all `-seg` variants and compute masks and boxes together, so `instance_segmentation` only selects which is stored: `segmentation_mask` annotations (mask *and* box) when ticked, `object_detection` (box only) when not.
- Model weights are downloaded on first use.
