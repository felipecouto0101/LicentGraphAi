import pytest

from scripts.export_audit_inventory import export


def test_cpu_mapping_keeps_actual_version_and_all_packages():
    inventory, requirements = export([("torch", "2.14.1+cpu"), ("ChromaDB", "1.5.9"), ("some_pkg", "2.0")])
    assert inventory == [
        {"name": "chromadb", "installed_version": "1.5.9", "audited_version": "1.5.9"},
        {"name": "some-pkg", "installed_version": "2.0", "audited_version": "2.0"},
        {"name": "torch", "installed_version": "2.14.1+cpu", "audited_version": "2.14.1"}]
    assert requirements == "chromadb==1.5.9\nsome-pkg==2.0\ntorch==2.14.1\n"


@pytest.mark.parametrize("packages", [[], [("torch", "2.14.1+custom")],
                                      [("chromadb", "1.5.9+custom")],
                                      [("ChromaDB", "1.5.9"), ("chromadb", "1.5.9")]])
def test_empty_duplicate_and_unknown_builds_fail(packages):
    with pytest.raises(ValueError):
        export(packages)
