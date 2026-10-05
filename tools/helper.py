"""Only the helper needed by the Gym environment."""
from collections.abc import Iterable


def flatten(items):
    """Recursively flatten nested iterables, except strings/bytes/dicts."""
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
            yield from flatten(item)
        else:
            yield item
