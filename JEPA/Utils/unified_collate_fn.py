from .jepa2data import tier2_collate_fn

def unified_collate_fn(batch):
    """
    Collate function for UnifiedDataset.
    Args:
        batch: list of dicts from UnifiedDataset.__getitem__()
            each dict has keys "j1" and/or "j2"
    Returns:
        dict with collated "j1" and "j2"
    """
    j1_list = [b["j1"] for b in batch if "j1" in b]
    j2_list = [b["j2"] for b in batch if "j2" in b]

    collated = {}

    # Collate JEPA-1: keep as list of tuples (like MapDataset returns)
    if j1_list:
        collated["j1"] = list(zip(*j1_list))

    # Collate JEPA-2 using the existing tier2_collate_fn
    if j2_list:
        collated["j2"] = tier2_collate_fn(j2_list)

    return collated
