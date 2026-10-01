"""Assess a complete pip-audit report without removing its original findings.

The CI must run tests/test_chroma_security.py before invoking this policy gate.
Known Chroma findings are reviewed for a narrowly enforced deployment, not
patched upstream. New findings, versions, available fixes or expired reviews fail.
"""
import argparse
from datetime import date
import json
from pathlib import Path


def assess(report, review, *, audit_exit_code, today=None):
    today = today or date.today()
    if audit_exit_code not in (0, 1):
        raise ValueError("pip-audit failed to complete")
    if review["scope"] != "local-rust-vector-only":
        raise ValueError("Unsupported review scope")
    if not date.fromisoformat(review["reviewed_on"]) <= today <= date.fromisoformat(review["expires_on"]):
        raise ValueError("Dependency review expired or has a future review date")
    dependencies = report["dependencies"]
    if not dependencies or any(d.get("skip_reason") for d in dependencies):
        raise ValueError("Dependency audit is empty or incomplete")
    chroma = [d for d in dependencies if d["name"] == review["package"]]
    if len(chroma) != 1 or chroma[0]["version"] != review["version"]:
        raise ValueError("Chroma version changed: a new applicability review is required")
    reviewed, blocking = [], []
    for dependency in dependencies:
        for finding in dependency.get("vulns", []):
            assessment = review["findings"].get(finding["id"])
            allowed = (
                dependency["name"] == review["package"]
                and dependency["version"] == review["version"]
                and assessment is not None
                and assessment["cve"] in finding.get("aliases", [])
                and not finding.get("fix_versions")
            )
            item = {"package": dependency["name"], "version": dependency["version"],
                    "id": finding["id"], "fix_versions": finding.get("fix_versions", [])}
            if allowed:
                item.update(status="not_applicable_to_enforced_deployment", reason=assessment["reason"])
                reviewed.append(item)
            else:
                item["status"] = "blocking"
                blocking.append(item)
    if (audit_exit_code == 0) != (not reviewed and not blocking):
        raise ValueError("pip-audit status does not match the report")
    return {"reviewed_findings": reviewed, "blocking_findings": blocking,
            "review_expires_on": review["expires_on"], "upstream_patched": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-exit-code", required=True, type=int)
    args = parser.parse_args()
    result = assess(json.loads(Path(args.report).read_text()),
                    json.loads(Path(args.review).read_text()), audit_exit_code=args.audit_exit_code)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for finding in result["reviewed_findings"]:
        print(f"REVIEWED {finding['id']}: {finding['reason']}")
    for finding in result["blocking_findings"]:
        print(f"BLOCKING {finding['package']} {finding['version']}: {finding['id']}")
    print("Raw findings are preserved; Chroma upstream vulnerabilities are not patched.")
    return 1 if result["blocking_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
