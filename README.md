<p align="center">
  <a href="https://lightly.ai/lightly-studio"> 
    <picture>
      <source
        media="(prefers-color-scheme: dark)"
        srcset="https://storage.googleapis.com/lightly-public/studio/lightlystudio_standard_horizontal_light.png"
      />
      <source
        media="(prefers-color-scheme: light)"
        srcset="https://storage.googleapis.com/lightly-public/studio/lightlystudio_standard_horizontal_dark.png"
      />
      <img
        src="https://storage.googleapis.com/lightly-public/studio/lightlystudio_standard_horizontal_dark.png"
        height="50"
        alt="LightlyStudio logo"
      />
    </picture>
  </a>
</p>
<p align="center"><strong>LightlyStudio Plugins</strong></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://docs.lightly.ai/studio/concepts_and_tools/plugins/"><img src="https://img.shields.io/badge/Docs-blue" alt="Documentation" /></a>
</p>

---
# Lightly Studio Plugins

A collection of installable plugins that extend the base functionality of [Lightly Studio](https://github.com/lightly-ai/lightly-studio).

Each plugin in this repository is packaged independently, installs in a single command, and is auto-discovered by Lightly Studio via Python entry points.

![SAM3 Plugin](plugins/sam3_segmentation/sam3_plugin.gif)

<p align="center"><em>SAM3 Plugin</em></p>

Each plugin entry below includes the exact copy-paste install command. After installation, the plugin is available in Lightly Studio automatically.

## Available Plugins

- [BBox auto propagation nano tracker](plugins/bbox_auto_propagation_nano_tracker/)  
  Propagates boxes from one annotated video frame to other frames in the same video.

  <details>
  <summary>Details</summary>

  If triggered from a frame, all bounding box annotations on that frame are
  propagated. If triggered from an annotation, only the selected annotation is
  propagated.

  - Scope: video only, within a single video
  - Entry points: frame or annotation
  - Controls: forward and backward propagation windows in seconds
  - Tradeoff: uses OpenCV NanoTracker, which is lightweight and fast on many
    machines but less robust on difficult motion, occlusion, or scale changes
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/bbox_auto_propagation_nano_tracker/`

  </details>

- [SAM3](plugins/sam3_segmentation/)  
  Segments all instances matching a text prompt in a single image or across the current view.

  <details>
  <summary>Details</summary>

  This is designed for dataset-wide prompt-based labeling workflows with
  class-like prompts such as `person`, `car`, or `dog`.

  - Scope: single image or images in the current view
  - Input: text prompt
  - Output: segmentation masks, or bounding boxes only
  - Labels: the prompt text is used as the annotation class name
  - Requirement: Hugging Face access to `facebook/sam3`
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/sam3_segmentation/`

  </details>

- [CLIP Zero-Shot Classification](plugins/clip_zero_shot_classification/)  
  Classifies images against a table of text prompts and the labels to assign, with no training.

  <details>
  <summary>Details</summary>

  You define the class vocabulary in the GUI as rows of `prompt` and `label`, so
  the label set is not fixed by a trained model. Each image is scored against
  every prompt and receives the label of the best match.

  - Scope: single image or images in the current view
  - Input: table of text prompts and the label each prompt assigns
  - Output: classification annotations, at most one per image
  - Labels: taken from the `label` column, defaulting to the prompt text, and
    created in the dataset if they do not exist yet
  - Recommended models:
    `openai/clip-vit-base-patch16` (default) for a speed/quality balance,
    `openai/clip-vit-base-patch32` for speed,
    `openai/clip-vit-large-patch14` for better accuracy
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/clip_zero_shot_classification/`

  </details>

- [LightlyTrain object detection inference](plugins/lightly_train_object_detection_inference/)  
  Runs LightlyTrain object detection inference on one image or the current view for auto-labeling.

  <details>
  <summary>Details</summary>

  You can use built-in LightlyTrain models for quick bootstrapping or provide a
  path to your own LightlyTrain checkpoint.

  - Scope: single image or images in the current view
  - Input: LightlyTrain model name or local path to a LightlyTrain checkpoint
  - Output: object detection annotations
  - Labels: class labels are read from the loaded model and created in the
    dataset if they do not exist yet
  - Recommended models:
    `dinov3/convnext-large-ltdetr-coco` for best performance,
    `dinov3/vits16-ltdetr-coco` for a speed/quality balance,
    `picodet-l-coco` for resource-constrained environments
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/lightly_train_object_detection_inference/`

  </details>

- [YOLO object detection](plugins/yolo_object_detection/)  
  Runs YOLO inference and adds bounding box annotations to unlabeled images.

  <details>
  <summary>Details</summary>

  Uses [Ultralytics YOLO](https://docs.ultralytics.com/) models for object detection auto-labeling.
  Supports any Ultralytics model name or a path to a custom checkpoint.

  - Scope: single image or images in the current view
  - Input: YOLO model name or local path to a YOLO checkpoint
  - Output: object detection annotations
  - Labels: class labels are read from the loaded model and created in the
    dataset if they do not exist yet
  - Recommended models:
    `yolov8n.pt` for speed, `yolov8s.pt` or `yolov8m.pt` for better accuracy
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/yolo_object_detection/`

  </details>

- [KITTI object detection export](plugins/kitti_export_object_detection/)  
  Exports KITTI object-detection label files.

  <details>
  <summary>Details</summary>

  The plugin writes KITTI `.txt` label files for the current filtered image
  view. Nested image folder structure is preserved in label filenames when
  exporting images from multiple folders.

  - Scope: images in the current view
  - Input: output folder
  - Output: KITTI object-detection label files
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/kitti_export_object_detection/`

  </details>

- [OpenRouter image captioning](plugins/openrouter_image_captioning/)  
  Captions a single image or the current view with a vision model served through OpenRouter.

  <details>
  <summary>Details</summary>

  Uses [OpenRouter](https://openrouter.ai/)'s OpenAI-compatible API, so any
  vision-capable model on the gateway can be used by changing one parameter.

  - Scope: single image or images in the current view
  - Input: model slug and prompt
  - Output: Lightly Studio captions
  - Requirement: an `OPENROUTER_API_KEY` environment variable
  - Default model: `qwen/qwen3-vl-8b-instruct`. Any vision-capable model from
    [openrouter.ai/models](https://openrouter.ai/models) works
  - Tradeoff: each run calls a paid API and blocks until it finishes, so filter the
    view down before captioning
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/openrouter_image_captioning/`

  </details>

## Contributing Plugins

1. Create a new directory under `plugins/`:
   ```
   plugins/my_plugin/
   ├── pyproject.toml
   └── src/lightly_plugins_my_plugin/
       ├── __init__.py
       └── operator.py
   ```

2. Register your operator class via entry points in `pyproject.toml`:
   ```toml
   [project.entry-points."lightly_studio.plugins"]
   my_plugin = "lightly_plugins_my_plugin.operator:MyPluginOperator"
   ```

3. Update `README.md` and `plugins.toml`.

4. Install: `pip install -e plugins/my_plugin`
