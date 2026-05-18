# Lightly Studio Plugins

A collection of installable plugins that extend the base functionality of [Lightly Studio](https://github.com/lightly-ai/lightly-studio).

Each plugin in this repository is packaged independently, installs in a single command, and is auto-discovered by Lightly Studio via Python entry points.

![SAM3 Segmentation Plugin](plugins/sam3_segmentation/sam3_plugin.gif)

<p align="center"><em>SAM3 Segmentation Plugin</em></p>

## Installing a Plugin

Choose a plugin from the table below and install it directly from this repository by pointing `pip` at the plugin subdirectory:

```bash
pip install "git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/<plugin-directory>"
```

After installation, the plugin is available in Lightly Studio automatically.

## Available Plugins

| Plugin | Description | Maintainer | Install |
|---|---|---|---|
| [BBox auto propagation nano tracker](plugins/bbox_auto_propagation_nano_tracker/)|Auto bbox propagation using nano tracker|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/bbox_auto_propagation_nano_tracker/`|
| [SAM3 Segmentation](plugins/sam3_segmentation/)|Automatic instance segmentation using SAM3 with a text prompt. Requires HuggingFace access to `facebook/sam3`.|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/sam3_segmentation/`|
| [LightlyTrain object detection inference](plugins/lightly_train_object_detection_inference/)|LightlyTrain inference operator for object detection auto-labeling|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/lightly_train_object_detection_inference/`|

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
