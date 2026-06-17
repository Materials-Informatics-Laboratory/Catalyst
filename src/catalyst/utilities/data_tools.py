import numpy as np

def parallel_sort(X,Y):
    # sort by values in y
    return [x for _, x in sorted(zip(Y, X), key=lambda pair: pair[0])], sorted(Y)

def make_pairs(list1, list2):
    return list(zip(list1, list2))

def remove_duplicate_tuples(tuple_list,return_unique_counts=True):
    seen = set()  # Use a set to keep track of seen tuples
    unique_list = []
    ids = []
    for i,tup in enumerate(tuple_list):
        if tup not in seen:
            unique_list.append(tup)
            ids.append(i)
            seen.add(tup)

    if return_unique_counts:
        return unique_list, ids
    else:
        return unique_list

def tuples_to_2d_lists(tuple_list):
    list1 = []
    list2 = []

    for tup in tuple_list:
        list1.append(tup[0])
        list2.append(tup[1])

    return [list1, list2]
def unique_lists_2d(lst,return_indices=1,sorted_search=1):
    unique_sublists = []
    indices = []
    if return_indices:
        for i, sublist in enumerate(lst):
            if sorted_search:
                sublist_tuple = tuple(sorted(sublist))
            else:
                sublist_tuple = tuple(sublist)
            if sublist_tuple not in unique_sublists:
                unique_sublists.append(sublist_tuple)
            indices.append(unique_sublists.index(sublist_tuple))
        return [list(sublist) for sublist in unique_sublists], indices
    else:
        for i, sublist in enumerate(lst):
            if sorted_search:
                sublist_tuple = tuple(sorted(sublist))
            else:
                sublist_tuple = tuple(sublist)
            if sublist_tuple not in unique_sublists:
                unique_sublists.append(sublist_tuple)
        return [list(sublist) for sublist in unique_sublists]

def remove_duplicate_list_pairs(list1, list2, stack=False):
    """
    Efficiently remove duplicate pairs (list1[i], list2[i]) from two parallel lists.
    Returns pair lists without duplicates and indices of unique pairs.

    Args:
        list1, list2: arrays or list of same length representing edges/pairs.
        stack: if True, returns np.vstack of the unique pairs; else returns lists.

    Returns:
        unique_pairs: tuple of lists or numpy arrays (unique edges)
        unique_ids: indices of unique pairs w.r.t original input
    """
    # Convert inputs to numpy arrays if not already
    arr1 = np.asarray(list1)
    arr2 = np.asarray(list2)

    # Stack pairs into 2D array
    pairs = np.stack((arr1, arr2), axis=-1)

    # Use np.unique with return_index to get unique pairs and their first occurrence indices
    unique_pairs, unique_ids = np.unique(pairs, axis=0, return_index=True)

    # Sort unique_ids to keep order consistent with input
    sorted_idx = np.argsort(unique_ids)
    unique_ids = unique_ids[sorted_idx]
    unique_pairs = unique_pairs[sorted_idx]

    if stack:
        return unique_pairs, unique_ids
    else:
        return unique_pairs[:, 0].tolist(), unique_pairs[:, 1].tolist(), unique_ids.tolist()
