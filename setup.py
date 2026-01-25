"""
Setup configuration for Concept Mapper.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = []
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                # Extract package name (before any comparison operators)
                pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
                requirements.append(pkg)
else:
    requirements = [
        "nltk>=3.9",
        "networkx>=3.2",
        "click>=8.1",
    ]

setup(
    name="concept-mapper",
    version="0.1.0",
    description="Extract and visualize conceptual vocabularies from texts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Concept Mapper Contributors",
    python_requires=">=3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "concept-mapper=concept_mapper.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
