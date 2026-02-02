#!/usr/bin/env python3
"""
Download required NLTK data for the cmapr project.

This script downloads all necessary NLTK datasets and models needed for:
- Tokenization (punkt)
- POS tagging (averaged_perceptron_tagger)
- Lemmatization (wordnet, omw-1.4)
- Reference corpus (brown)
- Stopwords (stopwords)
- Example corpus (movie_reviews)

Usage:
    python scripts/download_nltk_data.py

Or from Python:
    from scripts.download_nltk_data import download_all
    download_all()
"""

import nltk
import sys

# Required NLTK data packages
REQUIRED_PACKAGES = [
    # Tokenization
    ("punkt", "Punkt tokenizer models"),
    ("punkt_tab", "Punkt tokenizer tables"),
    # POS tagging
    ("averaged_perceptron_tagger", "Averaged perceptron POS tagger"),
    ("averaged_perceptron_tagger_eng", "English averaged perceptron POS tagger"),
    # Lemmatization
    ("wordnet", "WordNet lexical database"),
    ("omw-1.4", "Open Multilingual Wordnet"),
    # Reference corpora
    ("brown", "Brown corpus"),
    ("stopwords", "Stopwords corpus"),
    # Example/test corpora
    ("movie_reviews", "Movie reviews corpus"),
    ("gutenberg", "Gutenberg corpus"),
]


def download_package(package_name: str, description: str) -> bool:
    """
    Download a single NLTK package.

    Args:
        package_name: Name of the NLTK package
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading {package_name} ({description})...", end=" ")
        result = nltk.download(package_name, quiet=True)
        if result:
            print("✓")
            return True
        else:
            print("✗ (already up-to-date)")
            return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_all(verbose: bool = True) -> bool:
    """
    Download all required NLTK packages.

    Args:
        verbose: Print progress messages

    Returns:
        True if all downloads successful, False otherwise
    """
    if verbose:
        print("=" * 60)
        print("Downloading NLTK data for cmapr")
        print("=" * 60)
        print()

    failures = []

    for package_name, description in REQUIRED_PACKAGES:
        success = download_package(package_name, description)
        if not success:
            failures.append(package_name)

    print()
    if failures:
        print(f"✗ Failed to download {len(failures)} package(s):")
        for package in failures:
            print(f"  - {package}")
        return False
    else:
        print("✓ All NLTK data downloaded successfully!")
        return True


def verify_downloads() -> bool:
    """
    Verify that all required packages are available.

    Returns:
        True if all packages available, False otherwise
    """
    print("\nVerifying downloads...")

    # Test tokenization
    try:
        from nltk import word_tokenize, sent_tokenize

        test_text = "Hello, world! This is a test."
        word_tokenize(test_text)
        sent_tokenize(test_text)
        print("  ✓ Tokenization working")
    except Exception as e:
        print(f"  ✗ Tokenization failed: {e}")
        return False

    # Test POS tagging
    try:
        from nltk import pos_tag

        pos_tag(["test", "word"])
        print("  ✓ POS tagging working")
    except Exception as e:
        print(f"  ✗ POS tagging failed: {e}")
        return False

    # Test lemmatization
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("running", pos="v")
        print("  ✓ Lemmatization working")
    except Exception as e:
        print(f"  ✗ Lemmatization failed: {e}")
        return False

    # Test stopwords
    try:
        from nltk.corpus import stopwords

        stopwords.words("english")
        print("  ✓ Stopwords working")
    except Exception as e:
        print(f"  ✗ Stopwords failed: {e}")
        return False

    # Test Brown corpus
    try:
        from nltk.corpus import brown

        brown.words()[:10]
        print("  ✓ Brown corpus working")
    except Exception as e:
        print(f"  ✗ Brown corpus failed: {e}")
        return False

    print("\n✓ All verifications passed!")
    return True


def main():
    """Main entry point for the script."""
    success = download_all(verbose=True)

    if success:
        verify_success = verify_downloads()
        if verify_success:
            print("\n" + "=" * 60)
            print("Setup complete! NLTK data is ready to use.")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n⚠ Downloads completed but verification failed.")
            sys.exit(1)
    else:
        print(
            "\n⚠ Some downloads failed. Please check your internet connection and try again."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
