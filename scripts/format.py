#!/usr/bin/env python3
"""Format code using Black."""

import subprocess
import sys


def main():
    """Run Black formatter on Python files."""
    cmd = ["black", "*.py", "src/", "tests/", "scripts/"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
