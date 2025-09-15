from typing import Dict, List
from collections import defaultdict

def get_word_frequencies(doc: str) -> Dict[str, int]:
    """Extracts word frequencies from a document.

    Args:
        doc: Document content given as a string.

    Returns:
        Dictionary with words as keys and their frequencies as values.
    """
    # Punctuation marks to replace
    punctuation = [',', '.', ':', ';', '?', '!', '\n']

    # Convert the document to lowercase
    doc = doc.lower()

    # Replace each punctuation mark with a space
    for mark in punctuation:
        doc = doc.replace(mark, ' ')

    # Split the document into words based on whitespace
    tokens = doc.split()

    # Count the frequency of each token
    word_frequencies = defaultdict(int)
    for token in tokens:
        word_frequencies[token] += 1

    return dict(word_frequencies)


def get_word_feature_vector(
    word_frequencies: Dict[str, int], vocabulary: List[str]
) -> List[int]:
    """Creates a feature vector for a document, comprising word frequencies
        over a vocabulary.

    Args:
        word_frequencies: Dictionary with words as keys and frequencies as
            values.
        vocabulary: List of words.

    Returns:
        List of length `len(vocabulary)` with respective frequencies as values.
    """
    # Initialize the feature vector
    feature_vector = [0] * len(vocabulary)
    
    # Fill the vector
    for i, word in enumerate(vocabulary):
        if word in word_frequencies:
            feature_vector[i] = word_frequencies[word]
    
    return feature_vector
