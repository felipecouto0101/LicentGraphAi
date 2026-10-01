"""Preserve every audit finding; gate only vulnerabilities with available fixes."""
import argparse
import json
import os
from pathlib import Path


def assess(report, *, audit_exit_code):
    if audit_exit_code not in (0, 1):
        raise ValueError("pip-audit failed to complete")
    dependencies = report["dependencies"]
    if not dependencies or any(d.get("skip_reason") for d in dependencies):
        raise ValueError("Dependency audit is empty or incomplete")
    warnings, blocking = [], []
    for dependency in dependencies:
        for finding in dependency.get("vulns", []):
            fixes = finding["fix_versions"]
            if not isinstance(fixes, list) or any(not isinstance(v, str) or not v for v in fixes):
                raise ValueError("Invalid fixed-version information")
            item = {"package": dependency["name"], "version": dependency["version"],
                    "id": finding["id"], "fix_versions": fixes}
            if fixes:
                item["status"] = "blocking_fix_available"
                blocking.append(item)
            else:
                item.update(status="warning_no_fix_reported",
                            reason="No corrected version reported by the vulnerability database")
                warnings.append(item)
    if (audit_exit_code == 0) != (not warnings and not blocking):
        raise ValueError("pip-audit status does not match the report")
    return {"warning_findings": warnings, "blocking_findings": blocking,
            "policy": "block_available_fixes_warn_unfixed", "upstream_patched": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-exit-code", required=True, type=int)
    args = parser.parse_args()
    result = assess(json.loads(Path(args.report).read_text()), audit_exit_code=args.audit_exit_code)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["## Dependency audit policy", ""]
    for finding in result["warning_findings"]:
        message = f"WARNING {finding['package']} {finding['version']}: {finding['id']} (no fix reported)"
        print(message)
        lines.append(f"- {message}")
    for finding in result["blocking_findings"]:
        message = f"BLOCKING {finding['package']} {finding['version']}: {finding['id']}; fixes: {', '.join(finding['fix_versions'])}"
        print(message)
        lines.append(f"- {message}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")
    print("Raw findings are preserved; passing this policy does not mean vulnerabilities are fixed.")
    return 1 if result["blocking_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
