"""
Hyperparameter Schemas for EqProp Trainer

Defines model-specific hyperparameters that appear dynamically in the UI.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class HyperparamSpec:
    """Specification for a single hyperparameter."""
    name: str
    label: str
    type: str  # 'int', 'float', 'bool'
    default: Any
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    description: str = ""


# Registry of hyperparameter schemas by model/algorithm type
HYPERPARAMETER_SCHEMAS: Dict[str, List[HyperparamSpec]] = {
    # Standard EqProp models
    'eqprop': [
        HyperparamSpec(
            name='beta',
            label='Beta (Nudge Strength)',
            type='float',
            default=0.2,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
            description='Nudging strength for contrastive phase (0 = no nudging)'
        ),
        HyperparamSpec(
            name='eq_steps',
            label='Equilibrium Steps',
            type='int',
            default=30,
            min_val=5,
            max_val=100,
            step=5,
            description='Number of equilibrium settling iterations'
        ),
        HyperparamSpec(
            name='alpha',
            label='Alpha (Damping)',
            type='float',
            default=0.5,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            description='Damping factor for equilibrium updates'
        ),
    ],
    
    # Feedback Alignment variants
    'feedback_align': [
        HyperparamSpec(
            name='fa_scale',
            label='FA Feedback Scale',
            type='float',
            default=1.0,
            min_val=0.1,
            max_val=2.0,
            step=0.1,
            description='Scale of random feedback alignment matrix'
        ),
    ],
    
    # Adaptive FA
    'ada_fa': [
        HyperparamSpec(
            name='fa_scale',
            label='FA Feedback Scale',
            type='float',
            default=1.0,
            min_val=0.1,
            max_val=2.0,
            step=0.1,
            description='Initial scale of feedback matrix'
        ),
        HyperparamSpec(
            name='adapt_rate',
            label='Adaptation Rate',
            type='float',
            default=0.01,
            min_val=0.001,
            max_val=0.1,
            step=0.001,
            description='Rate of feedback matrix adaptation'
        ),
    ],
    
    # Equilibrium + Alignment hybrid
    'eq_align': [
        HyperparamSpec(
            name='beta',
            label='Beta (Nudge)',
            type='float',
            default=0.2,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
        ),
        HyperparamSpec(
            name='eq_steps',
            label='Eq Steps',
            type='int',
            default=20,
            min_val=5,
            max_val=50,
            step=5,
        ),
        HyperparamSpec(
            name='align_weight',
            label='Alignment Weight',
            type='float',
            default=0.5,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
            description='Weight for gradient alignment loss'
        ),
    ],
    
    # Momentum-based EqProp
    'mom_eq': [
        HyperparamSpec(
            name='beta',
            label='Beta',
            type='float',
            default=0.2,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
        ),
        HyperparamSpec(
            name='momentum',
            label='Momentum',
            type='float',
            default=0.9,
            min_val=0.0,
            max_val=0.99,
            step=0.01,
            description='Momentum coefficient for equilibrium updates'
        ),
    ],
    
    # Sparse EqProp
    'sparse_eq': [
        HyperparamSpec(
            name='beta',
            label='Beta',
            type='float',
            default=0.2,
            min_val=0.0,
            max_val=1.0,
            step=0.01,
        ),
        HyperparamSpec(
            name='sparsity',
            label='Sparsity Target',
            type='float',
            default=0.1,
            min_val=0.01,
            max_val=0.5,
            step=0.01,
            description='Target sparsity level (fraction of active neurons)'
        ),
    ],
    
    # Standard models (LoopedMLP, ConvEqProp, etc.)
    'looped_mlp': [
        HyperparamSpec(
            name='max_steps',
            label='Max Steps',
            type='int',
            default=30,
            min_val=5,
            max_val=100,
            step=5,
            description='Maximum equilibrium iterations'
        ),
    ],
    
    'conv_eqprop': [
        HyperparamSpec(
            name='gamma',
            label='Gamma (Damping)',
            type='float',
            default=0.5,
            min_val=0.1,
            max_val=1.0,
            step=0.05,
            description='Damping factor for convolutional layers'
        ),
    ],
    
    # Transformer LM variants
    'transformer_lm': [
        HyperparamSpec(
            name='eq_steps',
            label='Eq Steps',
            type='int',
            default=15,
            min_val=5,
            max_val=50,
            step=5,
        ),
        HyperparamSpec(
            name='alpha',
            label='Alpha',
            type='float',
            default=0.5,
            min_val=0.0,
            max_val=1.0,
            step=0.05,
        ),
    ],
}


def get_hyperparams_for_model(model_name: str) -> List[HyperparamSpec]:
    """
    Get hyperparameter specs for a given model name.
    
    Args:
        model_name: Model or algorithm name (can include description)
        
    Returns:
        List of HyperparamSpec objects
    """
    # Extract algorithm key from formatted names like "eqprop - Description"
    if ' - ' in model_name:
        key = model_name.split(' - ')[0].lower()
    else:
        key = model_name.lower().replace(' ', '_')
    
    # Map UI names to schema keys
    key_mappings = {
        'loopedmlp': 'looped_mlp',
        'conveqprop': 'conv_eqprop',
        'fulleqprop_transformer': 'transformer_lm',
        'attention-only_eqprop': 'transformer_lm',
        'recurrent_core_eqprop': 'transformer_lm',
        'hybrid_eqprop': 'transformer_lm',
        'loopedmlp_lm': 'looped_mlp',
        'backpropmlp_(baseline)': 'standard',
        'standardeqprop': 'eqprop',
    }
    
    key = key_mappings.get(key, key)
    
    return HYPERPARAMETER_SCHEMAS.get(key, [])


def hyperparams_to_dict(specs: List[HyperparamSpec]) -> Dict[str, Any]:
    """Convert list of specs to dict of default values."""
    return {spec.name: spec.default for spec in specs}
