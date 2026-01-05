import torch
import torch.nn as nn
import torch.nn.functional as F
from .equilibrium import EquilibriumSolver

class EqPropTrainer:
    """
    Implements the Equilibrium Propagation training loop.
    Phases:
    1. Free Phase: h* = relax(x)
    2. Nudged Phase: h_beta = relax(x, nudge_target)
    3. Update: delta_theta ~ (dE(h_beta) - dE(h*)) / beta
    """
    def __init__(self, model, optimizer, beta=0.22, alpha=0.5, epsilon=1e-4, max_steps=50,
                 update_strategy=None):
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.solver = EquilibriumSolver(epsilon, max_steps)
        self.criterion = nn.CrossEntropyLoss()
        self.update_strategy = update_strategy  # None = standard MSEProxyUpdate

    def step(self, x, y):
        self.optimizer.zero_grad()

        batch_size = x.size(0)
        
        # --- 1. FREE PHASE ---
        # Run dynamics to equilibrium without external force
        h_free, info_free = self.solver.solve(self.model, x)
        h_free = h_free.detach() # Don't backprop through time
        
        # Compute "gradients" for nudging?
        # Actually, for the nudged phase, we can use the explicit formulation:
        # h_{t+1} = ... - beta * dL/dh
        # We need dL/dh at the equilibrium point h_free.
        
        h_free_var = h_free.clone().requires_grad_(True)
        y_hat = self.model.Head(h_free_var)
        loss = self.criterion(y_hat, y)
        loss.backward()
        dL_dh = h_free_var.grad.detach()
        
        # --- 2. NUDGED PHASE ---
        # Initialize from h_free, apply nudging force
        # h_nudged = h_free - beta * dL_dh ? 
        # Or do we iterate again?
        # Scellier & Bengio (2017): Run dynamics again with clamping (nudging).
        # "One or few steps" often suffices if starting from h_free?
        # README says: "Repeat until ... (Nudge)"
        
        h_nudged, info_nudged = self.solver.solve(
            self.model, x, h_init=h_free, nudging=True, target_grads=dL_dh, beta=self.beta
        )
        h_nudged = h_nudged.detach()
        
        # --- 3. WEIGHT UPDATE (Contrastive) ---
        # Use update_strategy if provided (e.g., LocalHebbianUpdate for O(1) memory)
        if self.update_strategy is not None:
            # Use custom update strategy
            self.update_strategy.compute_update(
                self.model, h_free, h_nudged, x, y, self.optimizer
            )
        else:
            # Standard approach: backprop through energy function
            # Capture buffer from solver
            buffer_free = info_free.get('buffer', [])
            buffer_nudged = info_nudged.get('buffer', [])

            # Calculate Energy per state
            E_free = self.compute_energy(h_free, x, buffer_free)
            E_nudged = self.compute_energy(h_nudged, x, buffer_nudged)
            
            surrogate_loss = (E_nudged - E_free) / self.beta
            surrogate_loss.backward()
            
            self.optimizer.step()
        
        return {
            "loss": loss.item(), 
            "steps_free": info_free['steps'], 
            "steps_nudged": info_nudged['steps'],
            "converged_free": info_free.get('converged', False),
            "converged_nudged": info_nudged.get('converged', False)
        }
        
    def compute_energy(self, h, x, buffer=None):
        # We need the Energy E such that h_{t+1} approx h_t - grad_h E.
        # For h = tanh(Wx + Wh h), this corresponds to
        # E = 0.5*||h||^2 - sum( LogCosh(Wx + Wh h) ) ??
        # Or strictly Hopfield Energy E = -0.5 hWh ...
        #
        # Let's use the explicit Energy definition if possible, OR
        # use the "implicit" definition via the forward pass?
        #
        # "Vector Field" EqProp: We don't need explicit Energy if we use
        # (grad_h_t - grad_h_beta).
        #
        # Let's use the standard "Contrastive Signal" approach for general layers:
        # We run a "forward pass" with the stored h, and backprop from the node.
        #
        # Easier trick:
        # The update is roughly: h_post * h_pre' - h_post * h_pre'
        #
        # Let's stick to the surrogate loss Energy(h) if defined.
        # For LoopedMLP, we need to assert the energy form. 
        # If we assume symmetric weights, E = -0.5 hWh - hWx...
        #
        # Given we have Wx and Wh in the model.
        # E = 0.5 * ||h||^2 - (sum LogCosh(Wx + Wh h)) ? No, that gives h = tanh(...).
        # Wait, h = (1-a)h + a tanh(u). Fixed point h = tanh(u).
        #
        # Primitive function of tanh is LogCosh.
        # P(u) = Sum log cosh (u_i)
        # grad_u P = tanh(u) = h.
        # So if we define Scalar Potential P = Sum log cosh(Wx + Wh h).
        # Then grad_h P = Wh^T * tanh(...) = Wh^T * h (at fixed point).
        #
        # If W is symmetric, grad_h (0.5 hWh) = Wh.
        #
        # Let's implement the `energy` method on the model to keep this clean.
        
        return self.model.energy(h, x, buffer)

