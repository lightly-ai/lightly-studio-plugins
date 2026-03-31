"""Extract Lightly-maintained plugin directories from plugins.toml.

Outputs a JSON list suitable for use as a GitHub Actions matrix, e.g.:
  [{"plugin": "bbox_auto_propagation_nano_tracker", "path": "plugins/bbox_auto_propagation_nano_tracker"}]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import toml


def get_min_python_version(plugin_path: Path) -> str:
    """Read requires-python from the plugin's pyproject.toml and return the minimum version."""
    pyproject = plugin_path / "pyproject.toml"
    if not pyproject.exists():
        return "3.9"
    data = toml.load(pyproject)
    requires = data.get("project", {}).get("requires-python", ">=3.9")
    # Strip operator prefix (>=, ~=, ==, etc.) to get the bare version.
    version = requires.lstrip("><=!~^ ")
    return version


def get_lightly_plugins(repo_root: Path) -> list[dict[str, str]]:
    """Return a list of matrix entries for Lightly-maintained plugins."""
    plugins_toml = repo_root / "plugins.toml"
    data = toml.load(plugins_toml)

    matrix = []
    for plugin in data.get("plugins", []):
        if plugin.get("maintainer") != "lightly":
            continue
        source: str = plugin.get("source", "")
        if not source.startswith("local:"):
            continue
        rel_path = source.removeprefix("local:")
        plugin_dir = repo_root / rel_path
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

    return matrix


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    matrix = get_lightly_plugins(repo_root)
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()