import collections


def nested_dict():
    return collections.defaultdict(nested_dict)
