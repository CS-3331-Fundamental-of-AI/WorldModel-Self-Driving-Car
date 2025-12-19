# unified_dataset.py
from torch.utils.data import Dataset
from .jepa2data import tier2_collate_fn

class UnifiedDataset(Dataset):
    """
    Wrapper to unify multiple datasets (e.g., JEPA-1 and JEPA-2)
    without modifying their individual behavior.
    """
    def __init__(self, jepa1_dataset=None, jepa2_dataset=None):
        """
        Args:
            jepa1_dataset: instance of MapDataset (JEPA-1)
            jepa2_dataset: instance of Tier2Dataset (JEPA-2)
        """
        self.jepa1 = jepa1_dataset
        self.jepa2 = jepa2_dataset

        # Overall length = max of the individual datasets
        self.length = max(
            len(self.jepa1) if self.jepa1 else 0,
            len(self.jepa2) if self.jepa2 else 0
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns a dictionary with keys "j1" and/or "j2" depending on
        which datasets are provided.
        """
        out = {}

        if self.jepa1:
            out["j1"] = self.jepa1[idx % len(self.jepa1)]

        if self.jepa2:
            out["j2"] = self.jepa2[idx % len(self.jepa2)]

        return out


def unified_collate_fn(batch):
    """
    Collate function for UnifiedDataset.
    Preserves batch alignment across JEPA-1 / JEPA-2.
    """
    collated = {}

    # --- JEPA-1 ---
    j1_items = [b.get("j1", None) for b in batch]
    if any(x is not None for x in j1_items):
        # filter None, but keep relative order
        j1_valid = [x for x in j1_items if x is not None]
        collated["j1"] = list(zip(*j1_valid))
    else:
        collated["j1"] = None

    # --- JEPA-2 ---
    j2_items = [b.get("j2", None) for b in batch]
    if any(x is not None for x in j2_items):
        j2_valid = [x for x in j2_items if x is not None]
        collated["j2"] = tier2_collate_fn(j2_valid)
    else:
        collated["j2"] = None

    return collated
