import copy

import pytest

from scripts.check_dependency_audit import assess


def report(fixes=None, name="chromadb", version="1.5.9", advisory="PYSEC-2026-311"):
    return {"dependencies": [{"name": name, "version": version, "vulns": [
        {"id": advisory, "fix_versions": [] if fixes is None else fixes}]}]}


def test_unfixed_findings_are_warnings_and_report_is_preserved():
    source = report()
    original = copy.deepcopy(source)
    result = assess(source, audit_exit_code=1)
    assert len(result["warning_findings"]) == 1
    assert result["blocking_findings"] == []
    assert result["upstream_patched"] is False
    assert source == original


@pytest.mark.parametrize("package,version,advisory", [
    ("chromadb", "1.5.9", "NEW-ADVISORY"),
    ("other-package", "2.0", "CVE-NEW"),
    ("chromadb", "1.5.10", "PYSEC-2026-311")])
def test_no_fix_policy_does_not_depend_on_review_version_or_id(package, version, advisory):
    result = assess(report(name=package, version=version, advisory=advisory), audit_exit_code=1)
    assert result["warning_findings"] and not result["blocking_findings"]


def test_available_fix_blocks_even_for_chroma():
    result = assess(report(["1.5.10"]), audit_exit_code=1)
    assert result["blocking_findings"] and not result["warning_findings"]


def test_mixed_findings_keep_warning_and_block_fix():
    source = report()
    source["dependencies"].extend(report(["3.0"], name="another")["dependencies"])
    result = assess(source, audit_exit_code=1)
    assert len(result["warning_findings"]) == len(result["blocking_findings"]) == 1


def test_clean_report_passes():
    assert not assess({"dependencies": [{"name": "example", "version": "1", "vulns": []}]},
                      audit_exit_code=0)["blocking_findings"]


@pytest.mark.parametrize("change", ["audit_error", "skipped", "empty", "wrong_status", "missing_fixes", "bad_fixes"])
def test_failed_or_incomplete_audits_still_fail(change):
    source = report()
    status = 1
    if change == "audit_error":
        status = 2
    elif change == "skipped":
        source["dependencies"][0]["skip_reason"] = "Unknown version"
    elif change == "empty":
        source["dependencies"] = []
    elif change == "wrong_status":
        status = 0
    elif change == "missing_fixes":
        del source["dependencies"][0]["vulns"][0]["fix_versions"]
    else:
        source["dependencies"][0]["vulns"][0]["fix_versions"] = None
    with pytest.raises((ValueError, KeyError)):
        assess(source, audit_exit_code=status)
