# Security Policy

## Supported Versions

This project is currently in **Beta** (`Development Status :: 4 - Beta` on PyPI).
Security updates are provided for the latest released line.

| Version  | Supported          |
| -------- | ------------------ |
| 0.0.x    | :white_check_mark: |
| < 0.0.5  | :x:                |

Pre-release versions (anything below `0.0.5`) are not supported. Please upgrade
before reporting an issue.

## Reporting a Vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Instead, report privately via one of these channels:

1. **GitHub Security Advisories** (preferred): open a
   [private security advisory](https://github.com/pplmx/llm/security/advisories/new)
   on this repository. Only the maintainers will see it.
2. **Email**: contact the maintainer listed in `pyproject.toml` (`authors` field).

### What to include

- A clear description of the vulnerability and its impact.
- Steps to reproduce, ideally with a minimal example.
- Affected version(s) and commit SHA if known.
- Any known mitigations or workarounds.

### Response timeline

- **Acknowledgement**: within 7 days.
- **Triage & scope**: within 14 days.
- **Patch**: best-effort within 30 days, depending on severity and complexity.

If the report is declined (e.g., it is not actually a vulnerability, or it is in
unmaintained code), we will explain why.

## Disclosure Policy

We follow a **coordinated disclosure** model: please give us a reasonable window
to investigate and release a fix before any public write-up. We are happy to
credit reporters in the advisory unless they prefer to remain anonymous.
