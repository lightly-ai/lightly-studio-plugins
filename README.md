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

A collection of plugins for [Lightly Studio](https://github.com/lightly-ai/lightly-studio). Each plugin is independently pip-installable and auto-discovered via Python entry points.

| Plugin | Description | Maintainer | Install |
|---|---|---|---|
| [BBox auto propagation nano tracker](plugins/bbox_auto_propagation_nano_tracker/)|Auto bbox propagation using nano tracker|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/bbox_auto_propagation_nano_tracker/`|
| [SAM3 Segmentation](plugins/sam3_segmentation/)|Automatic instance segmentation using SAM3 with a text prompt. Requires HuggingFace access to `facebook/sam3`.|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/sam3_segmentation/`|
| [LightlyTrain object detection inference](plugins/lightly_train_object_detection_inference/)|LightlyTrain inference operator for object detection auto-labeling|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/lightly_train_object_detection_inference/`|

## Adding a New Plugin

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
