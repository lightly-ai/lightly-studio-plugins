# Lightly Studio Plugins

A collection of plugins for [Lightly Studio](https://github.com/lightly-ai/lightly-studio). Each plugin is independently pip-installable and auto-discovered via Python entry points.

| Plugin | Description | Maintainer | Install |
|---|---|---|---|
| [BBox auto propagation nano tracker](plugins/bbox_auto_propagation_nano_tracker/)|Auto bbox propagation using nano tracker|Lightly| `pip install git+https://github.com/lightly-ai/lightly-studio-plugins.git#subdirectory=plugins/bbox_auto_propagation_nano_tracker/`|

## Adding a New Plugin

1. Create a new directory under `plugins/`:
   ```
   plugins/my-plugin/
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

3. Update Readme.md and plugins.toml.

4. Install: `pip install -e plugins/my-plugin`
