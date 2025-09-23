from typing import List, Optional, Set, Tuple
from collections import defaultdict

PREFIX = "##"
UNKNOWN = "<unk>"


def initialize_vocabulary(word_corpus: List[str]) -> Set[str]:
    """Initializes the vocabulary with characters present in the corpus.

    Args:
        word_corpus: Corpus of words.

    Returns:
        Initial vocabulary.
    """
    vocab = set()
    for word in word_corpus:
        vocab.add(word[0])  # First character without the prefix
        for char in word[1:]:
            vocab.add(PREFIX + char)
    return vocab


def tokenize(word: str, vocabulary: Set[str]) -> List[str]:
    """Tokenizes a word using the vocabulary.

    The tokenizer splits the word using the longest possible tokens in the
    vocabulary. For example, if the word is "surfing", and the vocabulary
    contains the tokens "sur", "surf", and "ing", then the tokenizer will
    return ["surf", "##ing"].
    Returns <unk> token if the word cannot be fully tokenized.

    Args:
        word: Word to tokenize.
        vocabulary: Vocabulary.

    Returns:
        List of tokens.
    """
    word = word.lower()
    tokens = []
    while word.lstrip(PREFIX):
        for i in range(len(word)):
            if word[: len(word) - i] in vocabulary:
                tokens.append(word[: len(word) - i])
                word = PREFIX + word[len(word) - i :]
                break
        else:
            return [UNKNOWN]
    return tokens


def score(
    pair_freq: int, subword_token1_freq: int, subword_token2_freq: int
) -> float:
    """Calculates the score for merging two subword tokens.

    Args:
        pair_freq: Frequency of the pair.
        subword_token1_freq: Frequency of the first subword token.
        subword_token2_freq: Frequency of the second subword token.

    Returns:
        Score.
    """
    return pair_freq / (subword_token1_freq * subword_token2_freq)


def get_new_subword_token(
    data: List[Tuple[List[str], int]], vocabulary: Set[str]
) -> Tuple[str, float]:
    """Finds the new subword token to add to the vocabulary.

    The new subword token is the pair of tokens that maximizes the score. In
    case of ties, the pair that appears first in the vocabulary is chosen.

    Args:
        data: List of tokenized words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        New subword token and its score.
    """
    pair_counts = defaultdict(int)
    subword_freq = defaultdict(int)
    
    # Count frequencies of subword pairs and individual subwords
    for tokens, freq in data:
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
        for token in tokens:
            subword_freq[token] += freq
    
    # Find the pair with the highest score
    best_pair = None
    best_score = float('-inf')
    for (subword1, subword2), pair_freq in pair_counts.items():
        s = score(pair_freq, subword_freq[subword1], subword_freq[subword2])
        if s > best_score:
            best_score = s
            best_pair = (subword1, subword2)
    
    # Corrige el prefijo para el nuevo subword token
    new_token = best_pair[0] + best_pair[1].replace("##", "")
    
    return new_token, best_score


def train(
    word_corpus: List[Tuple[str, int]],
    vocabulary: Set[str],
    num_iterations: Optional[int] = 4,
    max_vocab_size: Optional[int] = None,
) -> Set[str]:
    """Executes the WordPiece training algorithm.

    The algorithm iteratively merges subword tokens to create new ones. It stops
    when the number of iterations is reached or when the vocabulary reaches
    the maximum size.

    Args:
        word_corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.
        num_iterations: Number of iterations to train the vocabulary. Defaults
            to 4.
        max_vocab_size: Maximum size of the vocabulary. Defaults to None.

    Returns:
        Vocabulary.
    """
    for _ in range(num_iterations):
        if max_vocab_size and len(vocabulary) >= max_vocab_size:
            break
        
        tokenized_corpus = tokenize_corpus(word_corpus, vocabulary)
        new_subword, _ = get_new_subword_token(tokenized_corpus, vocabulary)
        vocabulary.add(new_subword)
    
    return vocabulary


def tokenize_corpus(
    corpus: List[Tuple[str, int]], vocabulary: Set[str]
) -> List[Tuple[List[str], int]]:
    """Tokenizes the corpus using the vocabulary.

    Args:
        corpus: Corpus of words and their frequencies.
        vocabulary: Vocabulary.

    Returns:
        List of tokenized words and their frequencies.
    """
    tokenized_corpus = []
    for word, freq in corpus:
        tokens = tokenize(word, vocabulary)
        tokenized_corpus.append((tokens, freq))
    return tokenized_corpus
