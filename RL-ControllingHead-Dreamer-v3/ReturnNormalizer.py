
import torch
import DreamerConfig

class ReturnNormalizer:
    """Percentile-based return normalization for robust policy learning"""
    
    def __init__(self, config: DreamerConfig):
        self.decay = config.return_norm_decay
        self.limit = config.return_norm_limit
        self.low_percentile = config.return_percentile_low
        self.high_percentile = config.return_percentile_high
        self.scale = 1.0
    
    def update(self, returns: torch.Tensor):
        """Update scale using percentiles (robust to outliers)"""
        low = torch.quantile(returns, self.low_percentile / 100.0)
        high = torch.quantile(returns, self.high_percentile / 100.0)
        scale = high - low
        
        # Exponential moving average
        self.scale = self.decay * self.scale + (1 - self.decay) * scale.item()
    
    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        """Normalize returns, preserving small magnitudes"""
        scale = max(self.scale, self.limit)
        return returns / scale

