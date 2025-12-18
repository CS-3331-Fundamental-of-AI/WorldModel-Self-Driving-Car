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
            # transpose list of tuples into tuple of lists for batch
            collated["j1"] = list(zip(*j1_list))

        # Collate JEPA-2 using the existing tier2_collate_fn
        if j2_list:
            collated["j2"] = tier2_collate_fn(j2_list)

        return collated
