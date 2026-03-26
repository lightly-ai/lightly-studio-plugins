# Lightly Studio Plugins

A collection of plugins for [Lightly Studio](https://github.com/lightly-ai/lightly-studio). Each plugin is independently pip-installable and auto-discovered via Python entry points.

| Plugin | Description | Maintainer | Install |
|---|---|---|---|


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

3. Install: `pip install -e plugins/my-plugin`
