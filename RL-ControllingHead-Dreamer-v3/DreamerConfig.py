from dataclasses import dataclass

@dataclass
class DreamerConfig:
    """Configuration for DreamerV3 agent"""
    # Actor-Critic
    horizon: int = 15  # Imagination horizon (H)
    discount: float = 0.997  # γ (gamma)
    lambda_: float = 0.95  # λ for λ-returns
    entropy_scale: float = 3e-4  # η for exploration
    
    # Value learning
    value_loss_scale: float = 1.0  # β_val
    replay_value_loss_scale: float = 0.3  # β_repval
    value_ema_decay: float = 0.98  # For slow/target value network
    
    # Return normalization
    return_norm_decay: float = 0.99
    return_norm_limit: float = 1.0  # L - minimum normalization
    return_percentile_low: float = 5.0
    return_percentile_high: float = 95.0
    
    # Network architecture
    hidden_size: int = 512
    num_layers: int = 3
    activation: str = 'silu'
    
    # Training
    batch_size: int = 16
    sequence_length: int = 64
    learning_rate: float = 4e-5
    grad_clip: float = 100.0
    weight_decay: float = 0.0
