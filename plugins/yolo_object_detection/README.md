# YOLO Object Detection Plugin

Auto-labeling with [Ultralytics YOLO](https://docs.ultralytics.com/) models. Runs YOLO inference and adds bounding box annotations to images in Lightly Studio.

## Setup

### Install the plugin

```bash
uv pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yolo_object_detection/"
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `"yolov8n.pt"` | YOLO model weights path or Ultralytics model name (e.g. `yolov8s.pt`, `yolov8m.pt`, `/path/to/custom.pt`) |
| `confidence` | float | `0.25` | Minimum confidence threshold for keeping a prediction |
| `annotation_source` | string | `"yolo_auto_label__{model_path}"` | Target annotation source name where predictions will be stored |

## Notes

- Labels are read from the loaded model and created in the dataset if they do not exist yet.
- Annotations are stored in a collection named `yolo_auto_label__{model_path}` by default, configurable via `annotation_source`.
- Ultralytics model weights are downloaded automatically on first use.
