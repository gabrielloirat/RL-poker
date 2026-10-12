from collections.abc import Iterable


def flatten(items):

    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
            yield from flatten(item)
        else:
            yield item
