from typing import List


def get_confusion_matrix(
    actual: List[int], predicted: List[int]
) -> List[List[int]]:
    """Computes confusion matrix from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        List of two lists of length 2 each, representing the confusion matrix.
    """
    tn = fp = fn = tp = 0

    for a, p in zip(actual, predicted):
        if a == 0 and p == 0:
            tn += 1  # True Negative
        elif a == 0 and p == 1:
            fp += 1  # False Positive
        elif a == 1 and p == 0:
            fn += 1  # False Negative
        elif a == 1 and p == 1:
            tp += 1  # True Positive

    return [[tn, fp], [fn, tp]]


def accuracy(actual: List[int], predicted: List[int]) -> float:
    """Computes the accuracy from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Accuracy as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    total = tn + fp + fn + tp
    return (tn + tp) / total if total != 0 else 0.0


def precision(actual: List[int], predicted: List[int]) -> float:
    """Computes the precision from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Precision as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    return tp / (tp + fp) if (tp + fp) != 0 else 0.0


def recall(actual: List[int], predicted: List[int]) -> float:
    """Computes the recall from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        Recall as a float.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def f1(actual: List[int], predicted: List[int]) -> float:
    """Computes the F1-score from lists of actual or predicted labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of harmonic mean of precision and recall.
    """
    prec = precision(actual, predicted)
    rec = recall(actual, predicted)
    return (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0.0


def false_positive_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false positive rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as positive divided
            by number of actually negative instances.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    tn, fp = confusion_matrix[0]
    return fp / (fp + tn) if (fp + tn) != 0 else 0.0


def false_negative_rate(actual: List[int], predicted: List[int]) -> float:
    """Computes the false negative rate from lists of actual or predicted
        labels.

    Args:
        actual: List of integers (0 or 1) representing the actual classes of
            some instances.
        predicted: List of integers (0 or 1) representing the predicted classes
            of the corresponding instances.

    Returns:
        float of number of instances incorrectly classified as negative divided
            by number of actually positive instances.
    """
    confusion_matrix = get_confusion_matrix(actual, predicted)
    fn, tp = confusion_matrix[1]
    return fn / (fn + tp) if (fn + tp) != 0 else 0.0
