from typing import List, Union
from collections import Counter

def get_unique_elements(
    lst: List[Union[str, int]], n: int = 1
) -> List[Union[str, int]]:
    """
    Given a list of elements, returns those that repeat at least n times.
    The output list contains all unique elements, maintaining the order in which
    they first appear in the input list.

    Args:
        lst: Input list of elements.
        n (optional): Minimum number of times an element should repeat to be included.
                      Defaults to 1.

    Returns:
        List of unique elements that repeat at least n times.
    """
    # Count the occurrences of each element in the list
    element_count = Counter(lst)

    # Filter and return elements that appear at least n times, preserving order
    seen = set()
    result = []
    for element in lst:
        if element_count[element] >= n and element not in seen:
            result.append(element)
            seen.add(element)
    
    return result
