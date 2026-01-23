# Goal: produce a concept map of an input text.
# Respect structure of the text - parts, chapters, sections, paragraphs, etc.
# For each part get the key verbs, adverbs, nouns, adjectives.
#   Filter out common words.
# Output a data structure that can be used by D3 to make a concept map.

# Test to run in interactive shell.
# `python3`
# `import pos_tagger`
# `pos_tagger.run()`
# `import importlib`
# `importlib.reload(pos_tagger)`


def hi():
    print(f"hi")


def run():
    text = get_text()
    tokens = tokenize(text)
    res = pos_tag(tokens)
    return res


# POS tag all words.
import nltk
from nltk import word_tokenize, pos_tag

# # Download required NLTK data (uncomment if needed)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


# Read the text file
def get_text():
    with open("philosopher_1920_cc.txt", "r") as file:
        return file.read()


# Tokenize the text
def tokenize(text):
    return word_tokenize(text)


# Perform POS tagging
# tagged_words = pos_tag(tokens)

# Print the tagged words
# for word, tag in tagged_words:
#     print(f"{word}: {tag}")


# ##########
# Filter all, count, sort verbs.
# from nltk.probability import FreqDist

# def get_most_common_verbs(text, n):
#     # Download necessary NLTK data
#     # nltk.download('punkt', quiet=True)
#     # nltk.download('averaged_perceptron_tagger', quiet=True)

#     # Tokenize the text
#     tokens = word_tokenize(text)

#     # Perform part-of-speech tagging
#     tagged_words = pos_tag(tokens)

#     # Extract verbs (words tagged with VB, VBD, VBG, VBN, VBP, VBZ)
#     verbs = [word.lower() for word, tag in tagged_words if tag.startswith('V')]

#     # Calculate frequency distribution
#     freq_dist = FreqDist(verbs)

#     # Return the n most common verbs
#     return freq_dist.most_common(n)

# # Example usage
# # text =
# "Your input text goes here. Make sure it's a long text with various verbs."
# with open('philosopher_1920_cc.txt', 'r') as file:
#     text = file.read()

# n = 200  # Number of top verbs to retrieve

# top_verbs = get_most_common_verbs(text, n)

# for verb, count in top_verbs:
#     print(f"{verb}: {count}")

# #########
# Filter uncommon, count, sort verbs.
# from nltk import word_tokenize, pos_tag
# from nltk.probability import FreqDist
# from nltk.corpus import stopwords

# def get_content_rich_verbs(text, n):
#     # Download necessary NLTK data
#     nltk.download('punkt', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
#     nltk.download('stopwords', quiet=True)

#     # Tokenize the text
#     tokens = word_tokenize(text)

#     # Perform part-of-speech tagging
#     tagged_words = pos_tag(tokens)

#     # Define common verbs to filter out
#     common_verbs = set(['be', 'am', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
#                         'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should',
#                         'may', 'might', 'must', 'ought', 'go', 'get', 'make', 'know', 'think',
#                         'take', 'see', 'come', 'want', 'look', 'use', 'find', 'give', 'tell',
#                         'work', 'call', 'try'])

#     # Get English stopwords
#     stop_words = set(stopwords.words('english'))

#     # Extract verbs, excluding common verbs and stopwords
#     verbs = [word.lower() for word, tag in tagged_words
#              if tag.startswith('V') and word.lower() not in common_verbs and word.lower() not in stop_words]

#     # Calculate frequency distribution
#     freq_dist = FreqDist(verbs)

#     # Return the n most common content-rich verbs
#     return freq_dist.most_common(n)

# # Example usage
# with open('philosopher_1920_cc.txt', 'r') as file:
#     text = file.read()
# n = 10  # Number of top verbs to retrieve

# top_verbs = get_content_rich_verbs(text, n)
# for verb, count in top_verbs:
#     print(f"{verb}: {count}")

# ####
# Filter sentences with a given verb.

# from nltk import word_tokenize, pos_tag, sent_tokenize
# from nltk.probability import FreqDist
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# def get_text():
#     with open('philosopher_1920_cc.txt', 'r') as file:
#       text = file.read()
#     return text


# def get_verb_sentences(text, verb):
#     sentences = sent_tokenize(text)
#     verb_sentences = []
#     for sentence in sentences:
#         if verb in sentence.lower():
#             verb_sentences.append(sentence)

#     return verb_sentences


# def display(sentences):
#   print("\nSentences containing these verbs:")
#   for sentence in sentences:
#     print("-", sentence)


# def filter_sentences_with_verb(verb):
#   text = get_text()
#   sentences = get_verb_sentences(text, verb)
#   display(sentences)


# # filter_sentences_with_verb("becoming")
# filter_sentences_with_verb("transform")
