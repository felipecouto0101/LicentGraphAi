"""Export every installed distribution for a non-resolving vulnerability audit.

PyTorch CPU wheels use a local +cpu build suffix absent from the PyPI advisory
catalog. Match that one official build variant to the same public release; keep
the actual installed version and mapping in the inventory artifact.
"""
import argparse
from importlib.metadata import distributions
import json
from pathlib import Path

from packaging.utils import canonicalize_name
from packaging.version import Version


def export(packages):
    inventory = []
    seen = set()
    for name, installed_version in packages:
        name = canonicalize_name(name, validate=True)
        version = Version(installed_version)
        if name in seen:
            raise ValueError(f"Duplicate distribution: {name}")
        seen.add(name)
        if version.local and (name != "torch" or version.local != "cpu"):
            raise ValueError(f"Unreviewed local package build: {name} {version}")
        audited_version = version.public if version.local else str(version)
        inventory.append({"name": name, "installed_version": installed_version,
                          "audited_version": audited_version})
    if not inventory:
        raise ValueError("Empty package inventory")
    inventory.sort(key=lambda item: item["name"])
    requirements = "".join(f"{item['name']}=={item['audited_version']}\n" for item in inventory)
    return inventory, requirements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--requirements", required=True)
    args = parser.parse_args()
    inventory, requirements = export((d.metadata["Name"], d.version) for d in distributions())
    Path(args.inventory).write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    Path(args.requirements).write_text(requirements, encoding="utf-8")


if __name__ == "__main__":
    main()
