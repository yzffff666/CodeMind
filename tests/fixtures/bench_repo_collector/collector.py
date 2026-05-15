def append_item(item, bucket=[]):
    """Append one item and return only the items from this call."""
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
