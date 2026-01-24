#!/usr/bin/env python3
"""Lint code using Ruff."""

import subprocess
import sys


def main():
    """Run Ruff linter on Python files."""
    cmd = ["ruff", "check", "*.py", "src/", "tests/", "scripts/", "--fix"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
