import SymlogTransform
import torch
import torch.nn.functional as F


class TwoHotEncoder:
    """Two-hot encoding for distributional value prediction"""
    
    def __init__(self, num_bins: int = 255, vmin: float = -20, vmax: float = 20):
        self.num_bins = num_bins
        # Exponentially spaced bins in symlog space
        self.bins = torch.tensor(
            SymlogTransform.symexp(torch.linspace(vmin, vmax, num_bins))
        )
    
    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to two-hot encoding"""
        # Transform values to symlog space
        values_symlog = SymlogTransform.symlog(values)
        bins_symlog = SymlogTransform.symlog(self.bins.to(values.device))
        
        # Find the two closest bins
        below = (bins_symlog[None, :] <= values_symlog[..., None]).sum(-1) - 1
        below = torch.clamp(below, 0, self.num_bins - 2)
        above = below + 1
        
        # Linear interpolation weights
        below_val = bins_symlog[below]
        above_val = bins_symlog[above]
        weight_above = (values_symlog - below_val) / (above_val - below_val + 1e-8)
        weight_below = 1 - weight_above
        
        # Create two-hot encoding
        encoding = torch.zeros(*values.shape, self.num_bins, device=values.device)
        encoding.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
        encoding.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
        
        return encoding
    
    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to scalar values via expectation"""
        probs = F.softmax(logits, dim=-1)
        bins = self.bins.to(logits.device)
        return (probs * bins).sum(-1)


