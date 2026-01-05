import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import copy

@dataclass
class AttackResult:
    clean_acc: float
    adversarial_acc: float
    attack_success_rate: float
    avg_perturbation: float
    method: str
    epsilon: float

class AdversarialEvaluator:
    """Evaluates model robustness against adversarial attacks.
    
    Supports:
    - FGSM (Fast Gradient Sign Method)
    - PGD (Projected Gradient Descent)
    
    Uniquely handles EqProp's energy-based gradients vs BP's standard gradients.
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(self, dataloader, method: str = "pgd", epsilon: float = 0.1, 
                 steps: int = 10, alpha: float = 0.01) -> AttackResult:
        """Run robustness evaluation on a dataset."""
        correct_clean = 0
        correct_adv = 0
        total = 0
        total_perturbation = 0.0
        
        print(f"🛡️  Running {method.upper()} attack (ε={epsilon})...")
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)
            
            # Clean accuracy
            with torch.no_grad():
                outputs = self._forward(inputs)
                _, predicted = outputs.max(1)
                correct_clean += predicted.eq(targets).sum().item()
            
            # Generate adversarial examples
            if method == "fgsm":
                adv_inputs = self._fgsm_attack(inputs, targets, epsilon)
            elif method == "pgd":
                adv_inputs = self._pgd_attack(inputs, targets, epsilon, steps, alpha)
            else:
                raise ValueError(f"Unknown attack method: {method}")
            
            # Adversarial accuracy
            with torch.no_grad():
                adv_outputs = self._forward(adv_inputs)
                _, adv_predicted = adv_outputs.max(1)
                correct_adv += adv_predicted.eq(targets).sum().item()
            
            total += batch_size
            total_perturbation += (adv_inputs - inputs).abs().mean().item() * batch_size
            
        clean_acc = 100. * correct_clean / total
        adv_acc = 100. * correct_adv / total
        success_rate = 100. * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0.0
        
        return AttackResult(
            clean_acc=clean_acc,
            adversarial_acc=adv_acc,
            attack_success_rate=success_rate,
            avg_perturbation=total_perturbation / total,
            method=method,
            epsilon=epsilon
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Wrapper for model forward pass.
        
        Handles both Standard BP models and EqProp models (which usually need 
        an equilibrium phase).
        """
        if hasattr(self.model, "forward_with_equilibrium"):
            # EqProp model: Must run to equilibrium to get valid output
            # We assume the model has a method that does the full settling process
            # and returns the equilibrium state of the output layer.
            return self.model.forward_with_equilibrium(x)
        else:
            # Standard BP model
            return self.model(x)

    def _fgsm_attack(self, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Fast Gradient Sign Method."""
        inputs_adv = inputs.clone().detach()
        inputs_adv.requires_grad = True
        
        outputs = self._forward(inputs_adv)
        loss = F.cross_entropy(outputs, targets)
        
        self.model.zero_grad()
        loss.backward()
        
        # Collect the element-wise sign of the data gradient
        sign_data_grad = inputs_adv.grad.sign()
        
        # Create the perturbed image
        perturbed_inputs = inputs + epsilon * sign_data_grad
        
        # Clip to ensure valid pixel range [0, 1]
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        return perturbed_inputs.detach()

    def _pgd_attack(self, inputs: torch.Tensor, targets: torch.Tensor, epsilon: float, 
                    steps: int, alpha: float) -> torch.Tensor:
        """Projected Gradient Descent Attack."""
        inputs_adv = inputs.clone().detach()
        
        # Random initialization within epsilon ball
        inputs_adv = inputs_adv + torch.empty_like(inputs_adv).uniform_(-epsilon, epsilon)
        inputs_adv = torch.clamp(inputs_adv, 0, 1)
        
        for _ in range(steps):
            inputs_adv.requires_grad = True
            outputs = self._forward(inputs_adv)
            loss = F.cross_entropy(outputs, targets)
            
            self.model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                # Gradient update
                inputs_adv = inputs_adv + alpha * inputs_adv.grad.sign()
                
                # Projection: Clip perturbations to epsilon
                perturbation = torch.clamp(inputs_adv - inputs, -epsilon, epsilon)
                inputs_adv = torch.clamp(inputs + perturbation, 0, 1)
                
                inputs_adv.grad.zero_()
        
        return inputs_adv.detach()
