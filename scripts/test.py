#!/usr/bin/env python3
"""Run tests using pytest."""

import subprocess
import sys


def main():
    """Run pytest with verbose output."""
    cmd = ["pytest", "tests/", "-v"]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
