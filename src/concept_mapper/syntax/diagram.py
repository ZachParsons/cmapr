"""
Sentence diagramming and dependency parsing.

Uses Stanza for deep syntactic analysis to create dependency parse trees
and traditional sentence diagrams.
"""

from typing import List
import stanza
from pathlib import Path

# Global pipeline cache
_pipeline = None


def get_pipeline():
    """Get or initialize the Stanza pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = stanza.Pipeline(
            "en", processors="tokenize,pos,lemma,depparse", verbose=False
        )
    return _pipeline


def parse_sentence(text: str):
    """
    Parse a sentence using Stanza dependency parser.

    Args:
        text: The sentence to parse

    Returns:
        Stanza Document object with parsed sentences

    Example:
        >>> doc = parse_sentence("The cat sat on the mat.")
        >>> for sent in doc.sentences:
        ...     for word in sent.words:
        ...         print(f"{word.text}: {word.deprel}")
    """
    nlp = get_pipeline()
    return nlp(text)


def diagram_sentence(text: str, output_format: str = "ascii") -> str:
    """
    Create a sentence diagram from text.

    Args:
        text: The sentence to diagram
        output_format: Format for output ("ascii", "table", "tree")

    Returns:
        String representation of the diagram

    Example:
        >>> diagram = diagram_sentence("The question is what goal is envisaged.")
        >>> print(diagram)
    """
    doc = parse_sentence(text)

    output = []
    output.append("SENTENCE DIAGRAM - Dependency Parse")
    output.append("=" * 80)

    for sentence in doc.sentences:
        output.append(f"\nSentence: {sentence.text}\n")

        if output_format == "table":
            output.append(format_as_table(sentence))
        elif output_format == "tree":
            output.append(format_as_tree(sentence))
        else:  # ascii (default)
            output.append(format_as_table(sentence))
            output.append("\n\nDependency Tree:\n")
            output.append(format_as_tree(sentence))

    return "\n".join(output)


def format_as_table(sentence) -> str:
    """Format sentence parse as a table."""
    lines = []
    lines.append(f"{'ID':<5} {'Word':<20} {'POS':<8} {'Head':<5} {'Relation':<15}")
    lines.append("-" * 80)

    for word in sentence.words:
        lines.append(
            f"{word.id:<5} {word.text:<20} {word.pos:<8} "
            f"{word.head:<5} {word.deprel:<15}"
        )

    return "\n".join(lines)


def format_as_tree(sentence) -> str:
    """Format sentence parse as an ASCII tree."""
    # Find root
    root_words = [w for w in sentence.words if w.head == 0]
    if not root_words:
        return "No root found"

    root_word = root_words[0]
    lines = []

    def build_tree(word, indent=0, words_dict=None):
        if words_dict is None:
            words_dict = {w.id: w for w in sentence.words}

        # Add current word
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        lines.append(f"{prefix}{word.text} ({word.deprel})")

        # Add children
        children = sorted(
            [w for w in sentence.words if w.head == word.id], key=lambda w: w.id
        )
        for child in children:
            build_tree(child, indent + 1, words_dict)

    build_tree(root_word)
    return "\n".join(lines)


def print_dependency_tree(text: str):
    """
    Print a formatted dependency tree to console.

    Args:
        text: The sentence to analyze

    Example:
        >>> print_dependency_tree("The question is not what goal is envisaged.")
    """
    print(diagram_sentence(text, output_format="ascii"))


def save_diagram(text: str, output_path: Path, output_format: str = "ascii"):
    """
    Save sentence diagram to file.

    Args:
        text: The sentence to diagram
        output_path: Where to save the diagram
        output_format: Format ("ascii", "table", "tree")

    Example:
        >>> save_diagram(
        ...     "The cat sat on the mat.",
        ...     Path("output/diagram.txt")
        ... )
    """
    diagram = diagram_sentence(text, output_format)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(diagram)


def get_grammatical_relations(text: str) -> List[tuple]:
    """
    Extract all grammatical relations from a sentence.

    Args:
        text: The sentence to analyze

    Returns:
        List of (head, relation, dependent) tuples

    Example:
        >>> rels = get_grammatical_relations("The cat sat.")
        >>> for head, rel, dep in rels:
        ...     print(f"{head} --{rel}--> {dep}")
    """
    doc = parse_sentence(text)
    relations = []

    for sentence in doc.sentences:
        words_dict = {w.id: w for w in sentence.words}

        for word in sentence.words:
            if word.head == 0:
                head_text = "ROOT"
            else:
                head_text = words_dict[word.head].text

            relations.append((head_text, word.deprel, word.text))

    return relations
