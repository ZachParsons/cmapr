"""
Example usage of the refactored pos_tagger module.

Demonstrates how the pure, composable functions can be used in different ways.
"""

import pos_tagger as pt


def example_1_simple_analysis():
    """Example 1: Simple file analysis using convenience function."""
    print("=" * 60)
    print("Example 1: Simple File Analysis")
    print("=" * 60)

    result = pt.run("philosopher_1920_cc.txt")

    print(f"Total tokens: {result['token_count']}")
    print(f"\nTop 5 content verbs: {result['content_verbs'][:5]}")
    print(f"Top 5 nouns: {result['nouns'][:5]}")
    print(f"Top 5 adjectives: {result['adjectives'][:5]}")


def example_2_custom_pipeline():
    """Example 2: Build a custom pipeline by composing pure functions."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Pipeline (only verbs)")
    print("=" * 60)

    # Load and process step by step
    text = pt.load_text("philosopher_1920_cc.txt")
    tokens = pt.tokenize_words(text)
    tagged = pt.tag_parts_of_speech(tokens)

    # Extract only verbs, with custom filtering
    verbs = pt.extract_words_by_pos(tagged, "V")
    common_verbs = pt.get_common_verbs()
    stop_words = pt.get_stopwords_set()
    filtered = pt.filter_common_words(verbs, common_verbs, stop_words)
    freq_dist = pt.calculate_frequency_distribution(filtered)
    top_verbs = pt.get_most_common(freq_dist, 20)

    print("Top 20 content-rich verbs:")
    for verb, count in top_verbs:
        print(f"  {verb:20s} : {count}")


def example_3_search_and_analyze():
    """Example 3: Search for a term and analyze its context."""
    print("\n" + "=" * 60)
    print("Example 3: Search and Analyze Context")
    print("=" * 60)

    search_term = "consciousness"

    # Find sentences containing the term
    sentences = pt.search_term_in_file("philosopher_1920_cc.txt", search_term)
    print(f"Found {len(sentences)} sentences with '{search_term}'")

    # Analyze just those sentences
    combined_text = " ".join(sentences)
    result = pt.run_pipeline(combined_text, top_n=10)

    print(f"\nTop verbs co-occurring with '{search_term}':")
    for verb, count in result["content_verbs"]:
        print(f"  {verb}: {count}")

    print(f"\nTop nouns co-occurring with '{search_term}':")
    for noun, count in result["nouns"]:
        print(f"  {noun}: {count}")


def example_4_direct_text_analysis():
    """Example 4: Analyze text directly without a file."""
    print("\n" + "=" * 60)
    print("Example 4: Direct Text Analysis")
    print("=" * 60)

    sample_text = """
    The worker becomes alienated from the product of their labor.
    They cannot control what they produce or how it is used.
    This separation extends to their relationships with other workers,
    creating competition rather than cooperation.
    """

    result = pt.run_pipeline(sample_text, top_n=5)

    print(f"Tokens: {result['token_count']}")
    print(f"Content verbs: {result['content_verbs']}")
    print(f"Nouns: {result['nouns']}")
    print(f"Adjectives: {result['adjectives']}")


def example_5_compare_pos_categories():
    """Example 5: Compare different POS categories."""
    print("\n" + "=" * 60)
    print("Example 5: Compare POS Categories")
    print("=" * 60)

    text = pt.load_text("philosopher_1920_cc.txt")
    tokens = pt.tokenize_words(text)
    tagged = pt.tag_parts_of_speech(tokens)

    categories = {"Verbs": "V", "Nouns": "N", "Adjectives": "J", "Adverbs": "R"}

    for name, pos_prefix in categories.items():
        words = pt.extract_words_by_pos(tagged, pos_prefix)
        unique_words = len(set(words))
        total_words = len(words)
        print(f"{name:12s}: {total_words:5d} total, {unique_words:5d} unique")


if __name__ == "__main__":
    # Run all examples
    example_1_simple_analysis()
    example_2_custom_pipeline()
    example_3_search_and_analyze()
    example_4_direct_text_analysis()
    example_5_compare_pos_categories()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
