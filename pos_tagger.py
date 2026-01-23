"""
Concept Mapper: Extract key terms and structures from text for visualization.

Goal: Produce a concept map of an input text.
- Respect structure of the text - parts, chapters, sections, paragraphs, etc.
- For each part get the key verbs, adverbs, nouns, adjectives.
- Filter out common words.
- Output a data structure that can be used by D3 to make a concept map.

Usage:
    # Interactive shell
    python3
    import pos_tagger
    result = pos_tagger.run('philosopher_1920_cc.txt')

    # Or with custom text
    result = pos_tagger.run_pipeline("Your text here", top_n=10)
"""

from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from pathlib import Path
from typing import List, Tuple, Dict, Set

# ===== Core Functions (Pure, Composable) =====


def load_text(file_path: str) -> str:
    """Load text content from a file."""
    return Path(file_path).read_text(encoding="utf-8")


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into words."""
    return word_tokenize(text)


def tokenize_sentences(text: str) -> List[str]:
    """Tokenize text into sentences."""
    return sent_tokenize(text)


def tag_parts_of_speech(tokens: List[str]) -> List[Tuple[str, str]]:
    """Tag tokens with their parts of speech."""
    return pos_tag(tokens)


def extract_words_by_pos(
    tagged_tokens: List[Tuple[str, str]], pos_prefix: str
) -> List[str]:
    """
    Extract words matching a POS tag prefix.

    Args:
        tagged_tokens: List of (word, tag) tuples
        pos_prefix: POS tag prefix (e.g., 'V' for verbs, 'N' for nouns, 'J' for adjectives)

    Returns:
        List of lowercase words matching the POS prefix
    """
    return [word.lower() for word, tag in tagged_tokens if tag.startswith(pos_prefix)]


def get_common_verbs() -> Set[str]:
    """Return a set of common English verbs to filter out."""
    return {
        "be",
        "am",
        "is",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "ought",
        "go",
        "get",
        "make",
        "know",
        "think",
        "take",
        "see",
        "come",
        "want",
        "look",
        "use",
        "find",
        "give",
        "tell",
        "work",
        "call",
        "try",
    }


def get_stopwords_set() -> Set[str]:
    """Return English stopwords as a set."""
    return set(stopwords.words("english"))


def filter_common_words(
    words: List[str], common_words: Set[str], stop_words: Set[str]
) -> List[str]:
    """
    Filter out common words and stopwords.

    Args:
        words: List of words to filter
        common_words: Set of common words to exclude
        stop_words: Set of stopwords to exclude

    Returns:
        Filtered list of words
    """
    return [
        word for word in words if word not in common_words and word not in stop_words
    ]


def calculate_frequency_distribution(words: List[str]) -> FreqDist:
    """Calculate frequency distribution for a list of words."""
    return FreqDist(words)


def get_most_common(freq_dist: FreqDist, n: int) -> List[Tuple[str, int]]:
    """
    Get the n most common items from a frequency distribution.

    Args:
        freq_dist: NLTK FreqDist object
        n: Number of top items to return

    Returns:
        List of (word, count) tuples
    """
    return freq_dist.most_common(n)


def find_sentences_with_term(sentences: List[str], term: str) -> List[str]:
    """
    Find all sentences containing a specific term (case-insensitive).

    Args:
        sentences: List of sentences to search
        term: Term to search for

    Returns:
        List of sentences containing the term
    """
    return [sent for sent in sentences if term.lower() in sent.lower()]


# ===== Composed Pipeline Functions =====


def get_most_common_verbs(
    tagged_tokens: List[Tuple[str, str]], n: int
) -> List[Tuple[str, int]]:
    """
    Extract and count the most common verbs from tagged tokens.

    Args:
        tagged_tokens: List of (word, tag) tuples
        n: Number of top verbs to return

    Returns:
        List of (verb, count) tuples
    """
    verbs = extract_words_by_pos(tagged_tokens, "V")
    freq_dist = calculate_frequency_distribution(verbs)
    return get_most_common(freq_dist, n)


def get_content_rich_verbs(
    tagged_tokens: List[Tuple[str, str]], n: int
) -> List[Tuple[str, int]]:
    """
    Extract and count content-rich verbs (excluding common/auxiliary verbs).

    Args:
        tagged_tokens: List of (word, tag) tuples
        n: Number of top verbs to return

    Returns:
        List of (verb, count) tuples for content-rich verbs
    """
    verbs = extract_words_by_pos(tagged_tokens, "V")
    common_verbs = get_common_verbs()
    stop_words = get_stopwords_set()
    filtered_verbs = filter_common_words(verbs, common_verbs, stop_words)
    freq_dist = calculate_frequency_distribution(filtered_verbs)
    return get_most_common(freq_dist, n)


def get_content_rich_words_by_pos(
    tagged_tokens: List[Tuple[str, str]], pos_prefix: str, n: int
) -> List[Tuple[str, int]]:
    """
    Extract and count content-rich words for any POS category.

    Args:
        tagged_tokens: List of (word, tag) tuples
        pos_prefix: POS tag prefix ('V', 'N', 'J', 'R', etc.)
        n: Number of top words to return

    Returns:
        List of (word, count) tuples
    """
    words = extract_words_by_pos(tagged_tokens, pos_prefix)
    stop_words = get_stopwords_set()
    # For verbs, also filter common verbs
    if pos_prefix == "V":
        common_verbs = get_common_verbs()
        filtered_words = filter_common_words(words, common_verbs, stop_words)
    else:
        filtered_words = filter_common_words(words, set(), stop_words)

    freq_dist = calculate_frequency_distribution(filtered_words)
    return get_most_common(freq_dist, n)


# ===== Main Pipeline Functions =====


def analyze_text_file(file_path: str, top_n: int = 10) -> Dict:
    """
    Analyze a text file and extract key linguistic features.

    Args:
        file_path: Path to text file
        top_n: Number of top items to extract for each category

    Returns:
        Dictionary containing analysis results
    """
    text = load_text(file_path)
    return run_pipeline(text, top_n)


def run_pipeline(text: str, top_n: int = 10) -> Dict:
    """
    Run the complete analysis pipeline on text.

    Args:
        text: Text to analyze
        top_n: Number of top items to extract for each category

    Returns:
        Dictionary with keys: tokens, tagged, verbs, content_verbs, nouns, adjectives
    """
    tokens = tokenize_words(text)
    tagged = tag_parts_of_speech(tokens)

    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "tagged": tagged,
        "all_verbs": get_most_common_verbs(tagged, top_n),
        "content_verbs": get_content_rich_verbs(tagged, top_n),
        "nouns": get_content_rich_words_by_pos(tagged, "N", top_n),
        "adjectives": get_content_rich_words_by_pos(tagged, "J", top_n),
        "adverbs": get_content_rich_words_by_pos(tagged, "R", top_n),
    }


def search_term_in_file(file_path: str, term: str) -> List[str]:
    """
    Find all sentences containing a term in a file.

    Args:
        file_path: Path to text file
        term: Term to search for

    Returns:
        List of sentences containing the term
    """
    text = load_text(file_path)
    sentences = tokenize_sentences(text)
    return find_sentences_with_term(sentences, term)


# ===== Convenience function for backward compatibility =====


def run(file_path: str = "philosopher_1920_cc.txt") -> Dict:
    """
    Run analysis on a file (default: philosopher_1920_cc.txt).

    Args:
        file_path: Path to text file

    Returns:
        Dictionary containing analysis results
    """
    return analyze_text_file(file_path, top_n=10)
