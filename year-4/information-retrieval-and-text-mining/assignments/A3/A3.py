import re
import string
from typing import List, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from numpy import ndarray


def load_data(path: str) -> Tuple[List[str], List[int]]:
    """Loads data from file. Each except first (header) is a datapoint
    containing ID, Label, Email (content) separated by "\t". Lables should be
    changed into integers with 1 for "spam" and 0 for "ham".

    Args:
        path: Path to file from which to load data

    Returns:
        List of email contents and a list of lobels coresponding to each email.
    """
    contents = []
    labels = []
    
    with open(path, "r", encoding="utf-8") as file:
        next(file)  # Skip the header
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue  

            label = 1 if parts[1] == 'spam' else 0  
            email_content = parts[2].replace('""', '"') 
            
            if email_content.startswith('"') and email_content.endswith('"'):
                email_content = email_content[1:-1]
            
            contents.append(email_content)
            labels.append(label)
    
    return contents, labels


def preprocess(doc: str) -> str:
    """Preprocesses text to prepare it for feature extraction.

    Args:
        doc: String comprising the unprocessed contents of some email file.

    Returns:
        String comprising the corresponding preprocessed text.
    """
    doc = doc.lower() 
    doc = re.sub(f"[{re.escape(string.punctuation)}]", " ", doc) # Remove punctuation and special characters (except quotes)
    doc = re.sub(r'\d+', '', doc) # Remove numbers
    stopwords = set([
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "of", "in", "to", "on", "at", "for", "by", "with", "about", "as", "from", "that", "which",
        "who", "whom", "this", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
        "them", "my", "your", "his", "their", "our", "mine", "yours", "hers", "ours", "theirs"
        ])

    doc = ' '.join([word for word in doc.split() if word not in stopwords])
    return doc


def preprocess_multiple(docs: List[str]) -> List[str]:
    """Preprocesses multiple texts to prepare them for feature extraction.

    Args:
        docs: List of strings, each consisting of the unprocessed contents
            of some email file.

    Returns:
        List of strings, each comprising the corresponding preprocessed
            text.
    """
    return [preprocess(doc) for doc in docs]


def extract_features(
    train_dataset: List[str], test_dataset: List[str]
) -> Union[Tuple[ndarray, ndarray], Tuple[List[float], List[float]]]:
    """Extracts feature vectors from a preprocessed train and test datasets.

    Args:
        train_dataset: List of strings, each consisting of the preprocessed
            email content.
        test_dataset: List of strings, each consisting of the preprocessed
            email content.

    Returns:
        A tuple of of two lists. The lists contain extracted features for
          training and testing dataset respectively.
    """
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_dataset)
    X_test = vectorizer.transform(test_dataset) 
    return X_train, X_test


def train(X: ndarray, y: List[int]) -> object:
    """Trains a classifier on extracted feature vectors.

    Args:
        X: Numerical array-like object (2D) representing the instances.
        y: Numerical array-like object (1D) representing the labels.

    Returns:
        A trained model object capable of predicting over unseen sets of
            instances.
    """
    classifier = MultinomialNB()
    classifier.fit(X, y)
    return classifier


def evaluate(y: List[int], y_pred: List[int]) -> Tuple[float, float, float, float]:
    """Evaluates a model's predictive performance with respect to a labeled
    dataset.

    Args:
        y: Numerical array-like object (1D) representing the true labels.
        y_pred: Numerical array-like object (1D) representing the predicted
            labels.

    Returns:
        A tuple of four values: recall, precision, F_1, and accuracy.
    """
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    return recall, precision, f1, accuracy


if __name__ == "__main__":
    print("Loading data...")
    train_data_raw, train_labels = load_data("data/train.tsv")
    test_data_raw, test_labels = load_data("data/test.tsv")

    print("Processing data...")
    train_data = preprocess_multiple(train_data_raw)
    test_data = preprocess_multiple(test_data_raw)

    print("Extracting features...")
    train_feature_vectors, test_feature_vectors = extract_features(
        train_data, test_data
    )

    print("Training...")
    classifier = train(train_feature_vectors, train_labels)

    print("Applying model on test data...")
    predicted_labels = classifier.predict(test_feature_vectors)

    print("Evaluating...")
    recall, precision, f1, accuracy = evaluate(test_labels, predicted_labels)

    print(f"Recall:\t{recall}")
    print(f"Precision:\t{precision}")
    print(f"F1:\t{f1}")
    print(f"Accuracy:\t{accuracy}")
