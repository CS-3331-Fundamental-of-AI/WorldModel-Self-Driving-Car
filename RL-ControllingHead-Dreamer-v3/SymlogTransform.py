
import torch

class SymlogTransform:
    """Symlog transformation for robust value prediction across scales"""
    
    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        """symlog(x) = sign(x) * ln(|x| + 1)"""
        return torch.sign(x) * torch.log(torch.abs(x) + 1)
    
    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        """symexp(x) = sign(x) * (exp(|x|) - 1)"""
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
