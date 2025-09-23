from typing import Callable, Dict, List, Set

import ir_datasets


def load_rankings(
    filename: str = "data/system_rankings.tsv",
) -> Dict[str, List[str]]:
    """Load rankings from file. Every row in the file contains query ID and
    document ID separated by a tab ("\t").

        query_id    doc_id
        646	        4496d63c-8cf5-11e3-833c-33098f9e5267
        646	        ee82230c-f130-11e1-adc6-87dfa8eff430
        646	        ac6f0e3c-1e3c-11e3-94a2-6c66b668ea55

    Example return structure:

    {
        query_id_1: [doc_id_1, doc_id_2, ...],
        query_id_2: [doc_id_1, doc_id_2, ...]
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and list of documents as values.
    """
    rankings = {}
    with open(filename, "r") as file:
        # Skip the first line (it's a header)
        next(file)
        for line in file:
            query_id, doc_id = line.strip().split("\t")
            if query_id not in rankings:
                rankings[query_id] = []
            rankings[query_id].append(doc_id)
    return rankings


def load_ground_truth(
    collection: str = "wapo/v2/trec-core-2018",
) -> Dict[str, Set[str]]:
    """Load ground truth from ir_datasets. Qrel is a namedtuple class with
    following properties:

        query_id: str
        doc_id: str
        relevance: int
        iteration: str

    relevance is split into levels with values:

        0	not relevant
        1	relevant
        2	highly relevant

    This function considers documents to be relevant for relevance values
        1 and 2.

    Generic structure of returned dictionary:

    {
        query_id_1: {doc_id_1, doc_id_3, ...},
        query_id_2: {doc_id_1, doc_id_5, ...}
    }

    Args:
        filename (optional): Path to file with rankings. Defaults to
            "system_rankings.tsv".

    Returns:
        Dictionary with query IDs as keys and sets of documents as values.
    """
    ground_truth = {}
    dataset = ir_datasets.load(collection)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:  # Consider relevant if relevance is 1 or 2
            if qrel.query_id not in ground_truth:
                ground_truth[qrel.query_id] = set()
            ground_truth[qrel.query_id].add(qrel.doc_id)
    return ground_truth


def get_precision(
    system_ranking: List[str], ground_truth: Set[str], k: int = 100
) -> float:
    """Computes Precision@k.

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k: Cutoff. Only consider system rankings up to k.

    Returns:
        P@K (float).
    """
    if not system_ranking:
        return 0.0
    relevant_count = sum(1 for doc_id in system_ranking[:k] if doc_id in ground_truth)
    return relevant_count / k


def get_average_precision(
    system_ranking: List[str], ground_truth: Set[str]
) -> float:
    """Computes Average Precision (AP).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        AP (float).
    """
    if not system_ranking or not ground_truth:
        return 0.0
    ap_sum = 0.0
    relevant_count = 0
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            relevant_count += 1
            ap_sum += relevant_count / (i + 1)
    return ap_sum / len(ground_truth)


def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: Set[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
    """
    for i, doc_id in enumerate(system_ranking):
        if doc_id in ground_truth:
            return 1 / (i + 1)
    return 0.0


def get_mean_eval_measure(
    system_rankings: Dict[str, List[str]],
    ground_truths: Dict[str, Set[str]],
    eval_function: Callable,
) -> float:
    """Computes a mean of any evaluation measure over a set of queries.

    Args:
        system_rankings: Dict with query ID as key and a ranked list of
            document IDs as value.
        ground_truths: Dict with query ID as key and a set of relevant document
            IDs as value.
        eval_function: Callback function for the evaluation measure that mean
            is computed over.

    Returns:
        Mean evaluation measure (float).
    """
    total_score = 0.0
    count = 0
    for query_id, ranking in system_rankings.items():
        if query_id in ground_truths:
            score = eval_function(ranking, ground_truths[query_id])
            total_score += score
            count += 1
    return total_score / count if count > 0 else 0.0


if __name__ == "__main__":
    system_rankings = load_rankings()
    ground_truths = load_ground_truth()
