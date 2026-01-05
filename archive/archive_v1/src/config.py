"""Configuration dataclass for TorEqProp experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import argparse
import torch
from pathlib import Path
import yaml



@dataclass
class TorEqPropConfig:
    """Configuration for TorEqProp experiments.
    
    Automatically generates argparse CLI from dataclass fields.
    """
    # Dataset
    dataset: str = "mnist"  # mnist, cifar10, sst2
    batch_size: int = 128
    
    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    attention_type: str = "linear"  # softmax, linear
    symmetric: bool = False
    
    # Equilibrium solver
    max_iters: int = 50
    tol: float = 1e-5
    damping: float = 0.9
    
    # EqProp training
    beta: float = 0.1
    beta_anneal: bool = False  # Linear anneal β from 0.3 to 0.1
    lr: float = 1e-3
    epochs: int = 5
    update_mode: str = "mse_proxy"  # mse_proxy, vector_field
    
    # Performance
    compile: bool = False  # Use torch.compile
    seed: int = -1  # Random seed for reproducibility (-1 = no seed)
    rapid: bool = False  # Use rapid experimentation mode (loads configs/rapid_mode.yaml)
    
    # Logging
    wandb: bool = False
    wandb_project: str = "toreqprop"
    wandb_entity: Optional[str] = None
    log_interval: int = 100
    save_checkpoint: bool = False  # Disable checkpoint saving for fair timing
    
    # System
    device: str = ""  # Will be set in __post_init__
    num_workers: int = 4
    persistent_workers: bool = True  # Keep data loading workers alive between epochs
    
    def to_dict(self):
        """Convert config to dict for logging."""
        return asdict(self)
    
    @classmethod
    def from_args(cls):
        """Create config from command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Train TorEqProp models",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Automatically add all dataclass fields as arguments
        for field_name, field_info in cls.__dataclass_fields__.items():
            field_type = field_info.type
            default_val = field_info.default if field_info.default is not field_info.default_factory else field_info.default_factory()
            
            # Handle Optional types
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                field_type = field_type.__args__[0]
                default_val = None
            
            # For booleans, create store_true/store_false flags
            if field_type == bool:
                if default_val:
                    parser.add_argument(
                        f"--no-{field_name.replace('_', '-')}", 
                        dest=field_name,
                        action="store_false",
                        help=f"Disable {field_name}"
                    )
                else:
                    parser.add_argument(
                        f"--{field_name.replace('_', '-')}", 
                        action="store_true",
                        help=f"Enable {field_name}"
                    )
            else:
                # Handle Optional types - extract the actual type
                actual_type = field_type
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                    actual_type = field_type.__args__[0]
                
                parser.add_argument(
                    f"--{field_name.replace('_', '-')}", 
                    type=actual_type,
                    default=default_val,
                    help=f"{field_name}"
                )
        
        # First check if rapid mode is requested
        # We need to do this before final parsing to allow CLI args to override rapid defaults
        args_temp, _ = parser.parse_known_args()
        
        if args_temp.rapid:
            rapid_config_path = Path(__file__).parent.parent / "configs" / "rapid_mode.yaml"
            if rapid_config_path.exists():
                with open(rapid_config_path) as f:
                    rapid_defaults = yaml.safe_load(f)
                
                # Filter to only known fields and ensure types
                valid_defaults = {}
                for key, value in rapid_defaults.items():
                    if key in cls.__dataclass_fields__:
                        field_type = cls.__dataclass_fields__[key].type
                        if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                            field_type = field_type.__args__[0]
                        # Ensure type conversion
                        if field_type in (int, float, str, bool):
                            valid_defaults[key] = field_type(value)
                        else:
                            valid_defaults[key] = value
                
                print(f"[Rapid Mode] Loaded defaults from {rapid_config_path}")
                parser.set_defaults(**valid_defaults)
        
        # Now parse args with updated defaults
        args = parser.parse_args()
        config_dict = vars(args)
        
        return cls(**config_dict)
    
    def __post_init__(self):
        """Validate configuration."""
        # Set device if not specified
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate d_model divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # Validate symmetric mode requirements
        if self.symmetric and self.attention_type == "softmax":
            raise ValueError("Symmetric mode requires attention_type='linear'")
        
        # Set random seed if specified
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
