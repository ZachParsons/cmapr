"""
Inflection generation using inflect library and custom rules.

Maps Penn Treebank POS tags to inflected forms, handling nouns, verbs,
adjectives, and adverbs with common English inflection rules.
"""

from typing import Optional, Dict, List
import inflect


class InflectionGenerator:
    """
    Generate inflected forms from lemmas using POS tags.

    Uses inflect library for nouns and custom rules for verbs/adjectives.
    """

    # Irregular verb forms (most common cases)
    IRREGULAR_VERBS = {
        'be': {'VBD': 'was', 'VBN': 'been', 'VBG': 'being', 'VBZ': 'is', 'VBP': 'are'},
        'have': {'VBD': 'had', 'VBN': 'had', 'VBG': 'having', 'VBZ': 'has', 'VBP': 'have'},
        'do': {'VBD': 'did', 'VBN': 'done', 'VBG': 'doing', 'VBZ': 'does', 'VBP': 'do'},
        'go': {'VBD': 'went', 'VBN': 'gone', 'VBG': 'going', 'VBZ': 'goes', 'VBP': 'go'},
        'say': {'VBD': 'said', 'VBN': 'said', 'VBG': 'saying', 'VBZ': 'says', 'VBP': 'say'},
        'get': {'VBD': 'got', 'VBN': 'gotten', 'VBG': 'getting', 'VBZ': 'gets', 'VBP': 'get'},
        'make': {'VBD': 'made', 'VBN': 'made', 'VBG': 'making', 'VBZ': 'makes', 'VBP': 'make'},
        'know': {'VBD': 'knew', 'VBN': 'known', 'VBG': 'knowing', 'VBZ': 'knows', 'VBP': 'know'},
        'think': {'VBD': 'thought', 'VBN': 'thought', 'VBG': 'thinking', 'VBZ': 'thinks', 'VBP': 'think'},
        'take': {'VBD': 'took', 'VBN': 'taken', 'VBG': 'taking', 'VBZ': 'takes', 'VBP': 'take'},
        'see': {'VBD': 'saw', 'VBN': 'seen', 'VBG': 'seeing', 'VBZ': 'sees', 'VBP': 'see'},
        'come': {'VBD': 'came', 'VBN': 'come', 'VBG': 'coming', 'VBZ': 'comes', 'VBP': 'come'},
        'give': {'VBD': 'gave', 'VBN': 'given', 'VBG': 'giving', 'VBZ': 'gives', 'VBP': 'give'},
        'tell': {'VBD': 'told', 'VBN': 'told', 'VBG': 'telling', 'VBZ': 'tells', 'VBP': 'tell'},
        'run': {'VBD': 'ran', 'VBN': 'run', 'VBG': 'running', 'VBZ': 'runs', 'VBP': 'run'},
        'write': {'VBD': 'wrote', 'VBN': 'written', 'VBG': 'writing', 'VBZ': 'writes', 'VBP': 'write'},
        'eat': {'VBD': 'ate', 'VBN': 'eaten', 'VBG': 'eating', 'VBZ': 'eats', 'VBP': 'eat'},
        'find': {'VBD': 'found', 'VBN': 'found', 'VBG': 'finding', 'VBZ': 'finds', 'VBP': 'find'},
    }

    # Irregular adjective comparatives/superlatives
    IRREGULAR_ADJECTIVES = {
        'good': {'JJR': 'better', 'JJS': 'best'},
        'bad': {'JJR': 'worse', 'JJS': 'worst'},
        'far': {'JJR': 'farther', 'JJS': 'farthest'},
        'little': {'JJR': 'less', 'JJS': 'least'},
        'much': {'JJR': 'more', 'JJS': 'most'},
        'many': {'JJR': 'more', 'JJS': 'most'},
    }

    def __init__(self):
        self.inflector = inflect.engine()

    def inflect(self, lemma: str, pos_tag: str) -> str:
        """
        Generate inflected form of lemma for given POS tag.

        Args:
            lemma: Base form (e.g., "run", "good", "cat")
            pos_tag: Penn Treebank POS tag (e.g., "VBD", "JJR", "NNS")

        Returns:
            Inflected form (e.g., "ran", "better", "cats")
            Falls back to lemma if inflection unavailable

        Examples:
            >>> gen = InflectionGenerator()
            >>> gen.inflect("run", "VBD")
            'ran'
            >>> gen.inflect("good", "JJR")
            'better'
            >>> gen.inflect("cat", "NNS")
            'cats'
        """
        if not lemma:
            return lemma

        lemma_lower = lemma.lower()

        # Handle nouns
        if pos_tag in ('NN', 'NNP'):
            return lemma  # Singular - no change
        elif pos_tag in ('NNS', 'NNPS'):
            # Use inflect library for plurals
            plural = self.inflector.plural(lemma)
            return plural if plural else lemma + 's'

        # Handle verbs
        elif pos_tag.startswith('VB'):
            return self._inflect_verb(lemma_lower, pos_tag)

        # Handle adjectives
        elif pos_tag.startswith('JJ'):
            return self._inflect_adjective(lemma_lower, pos_tag)

        # Handle adverbs
        elif pos_tag.startswith('RB'):
            return self._inflect_adverb(lemma, pos_tag)

        # Unknown or unsupported POS tag
        return lemma

    def _inflect_verb(self, lemma: str, pos_tag: str) -> str:
        """Inflect verb using irregular forms or regular rules."""
        # Check irregular verbs first
        if lemma in self.IRREGULAR_VERBS:
            return self.IRREGULAR_VERBS[lemma].get(pos_tag, lemma)

        # Regular verb inflection rules
        if pos_tag == 'VB':
            return lemma  # Base form
        elif pos_tag == 'VBD' or pos_tag == 'VBN':
            # Past tense / past participle: add -ed
            if lemma.endswith('e'):
                return lemma + 'd'
            elif lemma.endswith('y') and len(lemma) > 1 and lemma[-2] not in 'aeiou':
                return lemma[:-1] + 'ied'
            elif self._should_double_consonant(lemma):
                return lemma + lemma[-1] + 'ed'
            else:
                return lemma + 'ed'
        elif pos_tag == 'VBG':
            # Present participle: add -ing
            if lemma.endswith('e') and not lemma.endswith('ee'):
                return lemma[:-1] + 'ing'
            elif lemma.endswith('ie'):
                return lemma[:-2] + 'ying'
            elif self._should_double_consonant(lemma):
                return lemma + lemma[-1] + 'ing'
            else:
                return lemma + 'ing'
        elif pos_tag == 'VBZ':
            # 3rd person singular: add -s
            if lemma.endswith(('s', 'x', 'z', 'ch', 'sh')):
                return lemma + 'es'
            elif lemma.endswith('y') and len(lemma) > 1 and lemma[-2] not in 'aeiou':
                return lemma[:-1] + 'ies'
            else:
                return lemma + 's'
        elif pos_tag == 'VBP':
            return lemma  # Present non-3rd person (same as base)

        return lemma

    def _inflect_adjective(self, lemma: str, pos_tag: str) -> str:
        """Inflect adjective (comparative/superlative)."""
        # Check irregular adjectives
        if lemma in self.IRREGULAR_ADJECTIVES:
            return self.IRREGULAR_ADJECTIVES[lemma].get(pos_tag, lemma)

        if pos_tag == 'JJ':
            return lemma  # Base form
        elif pos_tag == 'JJR':
            # Comparative: add -er
            if lemma.endswith('e'):
                return lemma + 'r'
            elif lemma.endswith('y') and len(lemma) > 1:
                return lemma[:-1] + 'ier'
            elif self._should_double_consonant(lemma):
                return lemma + lemma[-1] + 'er'
            else:
                return lemma + 'er'
        elif pos_tag == 'JJS':
            # Superlative: add -est
            if lemma.endswith('e'):
                return lemma + 'st'
            elif lemma.endswith('y') and len(lemma) > 1:
                return lemma[:-1] + 'iest'
            elif self._should_double_consonant(lemma):
                return lemma + lemma[-1] + 'est'
            else:
                return lemma + 'est'

        return lemma

    def _inflect_adverb(self, lemma: str, pos_tag: str) -> str:
        """Inflect adverb (many use 'more'/'most' instead of -er/-est)."""
        if pos_tag == 'RB':
            return lemma

        # Most adverbs, especially -ly adverbs, use more/most
        # Only simple adverbs have -er/-est forms
        # For now, return unchanged (correct behavior for most adverbs)
        return lemma

    def _should_double_consonant(self, word: str) -> bool:
        """
        Check if final consonant should be doubled before suffix.

        Rules: CVC pattern (consonant-vowel-consonant) in stressed syllable
        Examples: run -> running, big -> bigger
        """
        if len(word) < 2:
            return False

        # Simple heuristic: single syllable words ending in CVC
        if len(word) <= 3 and word[-1] not in 'aeiouwyxz' and word[-2] in 'aeiou':
            if len(word) == 3:
                return word[-3] not in 'aeiou'
            return True

        return False

    def get_all_forms(self, lemma: str) -> Dict[str, List[str]]:
        """
        Get all possible inflected forms for a lemma.

        Args:
            lemma: Base form to inflect

        Returns:
            Dictionary mapping POS tags to list of inflected forms
        """
        forms = {}

        # Noun forms
        forms['NN'] = [lemma]
        forms['NNS'] = [self.inflect(lemma, 'NNS')]

        # Verb forms
        for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            forms[tag] = [self.inflect(lemma, tag)]

        # Adjective forms
        for tag in ['JJ', 'JJR', 'JJS']:
            forms[tag] = [self.inflect(lemma, tag)]

        return forms

    def can_inflect(self, lemma: str, pos_tag: str) -> bool:
        """
        Check if a lemma can be inflected for given POS tag.

        Args:
            lemma: Base form to check
            pos_tag: Penn Treebank POS tag

        Returns:
            True if inflection is supported
        """
        supported_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
            'JJ', 'JJR', 'JJS',
            'RB', 'RBR', 'RBS'
        }
        return pos_tag in supported_tags
