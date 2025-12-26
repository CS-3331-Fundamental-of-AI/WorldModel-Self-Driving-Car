# JEPA_SecondLayer/__init__.py

from .inverse_affordance.inverse_affordance import JEPA_Tier2_InverseAffordance
from .physical_affordance import JEPA_Tier2_PhysicalAffordance

__all__ = [
    "JEPA_Tier2_InverseAffordance",
    "JEPA_Tier2_PhysicalAffordance",
]
