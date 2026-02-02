#!/usr/bin/env python3
"""Format code using Ruff."""

import subprocess
import sys


def main():
    """Run Ruff formatter on Python files."""
    cmd = ["ruff", "format", "*.py", "src/", "tests/", "scripts/"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
