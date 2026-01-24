#!/usr/bin/env python3
"""Run all checks: format, lint, and test."""

import subprocess
import sys


def main():
    """Run Black, Ruff, and pytest in sequence."""
    print("🔧 Running Black formatter...")
    result = subprocess.run(["black", "*.py", "src/", "tests/", "scripts/"])
    if result.returncode != 0:
        print("❌ Black formatting failed")
        sys.exit(result.returncode)

    print("\n🔍 Running Ruff linter...")
    result = subprocess.run(["ruff", "check", "*.py", "src/", "tests/", "scripts/"])
    if result.returncode != 0:
        print("❌ Ruff linting failed")
        sys.exit(result.returncode)

    print("\n🧪 Running tests...")
    result = subprocess.run(["pytest", "tests/", "-v"])
    if result.returncode != 0:
        print("❌ Tests failed")
        sys.exit(result.returncode)

    print("\n✅ All checks passed! Ready to commit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
