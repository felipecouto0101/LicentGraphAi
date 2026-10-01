import copy
from datetime import date
import json
from pathlib import Path

import pytest

from scripts.check_dependency_audit import assess


def inputs():
    review = json.loads((Path(__file__).parents[1] / "security/dependency-reviews.json").read_text())
    report = {"dependencies": [{"name": "chromadb", "version": "1.5.9", "vulns": [
        {"id": name, "aliases": [value["cve"]], "fix_versions": []}
        for name, value in review["findings"].items()]}]}
    return report, review


def run(report, review, status=1, today=date(2026, 10, 1)):
    return assess(report, review, audit_exit_code=status, today=today)


def test_only_exact_reviewed_findings_are_accepted_and_retained():
    report, review = inputs()
    original = copy.deepcopy(report)
    result = run(report, review)
    assert len(result["reviewed_findings"]) == 4
    assert result["blocking_findings"] == []
    assert result["upstream_patched"] is False
    assert report == original


@pytest.mark.parametrize("change", ["new_id", "new_package", "available_fix", "wrong_cve"])
def test_other_findings_still_block(change):
    report, review = inputs()
    dependency = report["dependencies"][0]
    if change == "new_id":
        dependency["vulns"][0]["id"] = "CVE-NEW"
    elif change == "new_package":
        other = copy.deepcopy(dependency)
        other["name"] = "different-package"
        report["dependencies"].append(other)
    elif change == "available_fix":
        dependency["vulns"][0]["fix_versions"] = ["1.5.10"]
    else:
        dependency["vulns"][0]["aliases"] = ["CVE-WRONG"]
    assert run(report, review)["blocking_findings"]


@pytest.mark.parametrize("change", ["version", "expired", "audit_error", "skipped", "empty", "wrong_status"])
def test_incomplete_or_stale_assessment_fails(change):
    report, review = inputs()
    status, today = 1, date(2026, 10, 1)
    if change == "version":
        report["dependencies"][0]["version"] = "1.5.10"
    elif change == "expired":
        today = date(2026, 11, 1)
    elif change == "audit_error":
        status = 2
    elif change == "skipped":
        report["dependencies"][0]["skip_reason"] = "Unknown version"
    elif change == "empty":
        report["dependencies"] = []
    else:
        status = 0
    with pytest.raises(ValueError):
        run(report, review, status=status, today=today)
