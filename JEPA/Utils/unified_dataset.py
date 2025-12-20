# unified_dataset.py
from torch.utils.data import Dataset
from .jepa2data import tier2_collate_fn
from .jepa3data import tier3_collate_fn   

class UnifiedDataset(Dataset):
    """
    Wrapper to unify JEPA-1 / JEPA-2 / JEPA-3 datasets.
    """

    def __init__(self, jepa1_dataset=None, jepa2_dataset=None, jepa3_dataset=None):
        self.jepa1 = jepa1_dataset
        self.jepa2 = jepa2_dataset
        self.jepa3 = jepa3_dataset

        self.length = max(
            len(self.jepa1) if self.jepa1 else 0,
            len(self.jepa2) if self.jepa2 else 0,
            len(self.jepa3) if self.jepa3 else 0,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        out = {}

        if self.jepa1:
            out["j1"] = self.jepa1[idx % len(self.jepa1)]

        if self.jepa2:
            out["j2"] = self.jepa2[idx % len(self.jepa2)]

        if self.jepa3:
            out["j3"] = self.jepa3[idx % len(self.jepa3)]

        return out

def unified_collate_fn(batch):
    collated = {}

    # --------------------
    # JEPA-1
    # --------------------
    j1_items = [b.get("j1", None) for b in batch]
    if any(x is not None for x in j1_items):
        j1_valid = [x for x in j1_items if x is not None]
        collated["j1"] = list(zip(*j1_valid))
    else:
        collated["j1"] = None

    # --------------------
    # JEPA-2
    # --------------------
    j2_items = [b.get("j2", None) for b in batch]
    if any(x is not None for x in j2_items):
        j2_valid = [x for x in j2_items if x is not None]
        collated["j2"] = tier2_collate_fn(j2_valid)
    else:
        collated["j2"] = None

    # --------------------
    # JEPA-3  
    # --------------------
    j3_items = [b.get("j3", None) for b in batch]
    if any(x is not None for x in j3_items):
        j3_valid = [x for x in j3_items if x is not None]
        collated["j3"] = tier3_collate_fn(j3_valid)
    else:
        collated["j3"] = None

    return collated
