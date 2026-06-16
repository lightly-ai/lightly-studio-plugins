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

![SAM3 Segmentation Plugin](plugins/sam3_segmentation/sam3_plugin.gif)

<p align="center"><em>SAM3 Segmentation Plugin</em></p>

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

- [SAM3 Segmentation](plugins/sam3_segmentation/)  
  Segments all instances matching a text prompt in a single image or across the current view.

  <details>
  <summary>Details</summary>

  This is designed for dataset-wide prompt-based labeling workflows with
  class-like prompts such as `person`, `car`, or `dog`.

  - Scope: single image or images in the current view
  - Input: text prompt
  - Output: segmentation masks
  - Labels: the prompt text is used as the annotation class name
  - Requirement: Hugging Face access to `facebook/sam3`
  - Maintainer: Lightly
  - Install:
    `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/sam3_segmentation/`

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
