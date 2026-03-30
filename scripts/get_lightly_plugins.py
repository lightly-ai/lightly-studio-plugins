"""Extract Lightly-maintained plugin directories from plugins.toml.

Outputs a JSON list suitable for use as a GitHub Actions matrix, e.g.:
  [{"plugin": "bbox_auto_propagation_nano_tracker", "path": "plugins/bbox_auto_propagation_nano_tracker"}]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGINS_TOML = REPO_ROOT / "plugins.toml"


def get_min_python_version(plugin_path: Path) -> str:
    """Read requires-python from the plugin's pyproject.toml and return the minimum version."""
    pyproject = plugin_path / "pyproject.toml"
    if not pyproject.exists():
        return "3.9"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    requires = data.get("project", {}).get("requires-python", ">=3.9")
    # Strip operator prefix (>=, ~=, ==, etc.) to get the bare version.
    version = requires.lstrip("><=!~^ ")
    return version


def main() -> None:
    with open(PLUGINS_TOML, "rb") as f:
        data = tomllib.load(f)

    matrix = []
    for plugin in data.get("plugins", []):
        if plugin.get("maintainer") != "lightly":
            continue
        source: str = plugin.get("source", "")
        if not source.startswith("local:"):
            continue
        rel_path = source.removeprefix("local:")
        plugin_dir = REPO_ROOT / rel_path
        if not plugin_dir.is_dir():
            print(f"WARNING: plugin directory not found: {plugin_dir}", file=sys.stderr)
            continue
        matrix.append(
            {
                "plugin": plugin_dir.name,
                "path": rel_path,
                "min_python": get_min_python_version(plugin_dir),
            }
        )

    print(json.dumps(matrix))
