import copy


def _shorten(emb, head, tail):
    return emb[:head] + ['...'] + emb[-tail:] if len(emb) > head + tail else emb


def summarize_embeddings(data, head=1, tail=1):
    """Returns a copy of data with long embedding lists truncated for printing.

    Accepts a single vector, a list of vectors, or a list of dicts with an 'embedding' key.
    """
    data_copy = copy.deepcopy(data)
    if data_copy and all(isinstance(x, (int, float)) for x in data_copy):
        return _shorten(data_copy, head, tail)
    for i, item in enumerate(data_copy):
        if isinstance(item, dict) and 'embedding' in item:
            item['embedding'] = _shorten(item['embedding'], head, tail)
        elif isinstance(item, list):
            data_copy[i] = _shorten(item, head, tail)
    return data_copy
