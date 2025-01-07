
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