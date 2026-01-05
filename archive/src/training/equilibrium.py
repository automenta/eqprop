import torch

class EquilibriumSolver:
    """
    Manages the equilibrium propagation dynamics.
    Finds the fixed point h* such that h_{t+1} approx h_t.
    """
    def __init__(self, epsilon=1e-4, max_steps=50):
        self.epsilon = epsilon
        self.max_steps = max_steps

    def solve(self, model, x, h_init=None, nudging=False, target_grads=None, beta=0.0):
        """
        Run dynamics until convergence.
        
        Args:
            model: The Looped/Toroidal model.
            x: Input.
            h_init: Initial state (optional).
            nudging: If True, apply nudging.
            target_grads: Gradient of Loss w.r.t h (for nudging).
            beta: Nudging strength.
        
        Returns:
            h_star: Converting state.
            info: Dict with steps taken, residual.
        """
        # We need a unified interface for model.forward_step
        # Assuming model has forward_step(h, x, ...)
        
        batch_size = x.size(0)
        
        if h_init is None:
            if hasattr(model, 'init_state'):
                h = model.init_state(x)
            else:
                h = torch.zeros(batch_size, model.hidden_dim, device=x.device)
        else:
            h = h_init.clone()

        # Handle buffer state 
        # (For rigorous TEP, we should pass buffer_init, but for now init to None matches forward)
        buffer_state = None 

        with torch.no_grad():
            for t in range(self.max_steps):
                h_prev = h
                
                # Generic step: inputs h, x, prev_buffer -> outputs h_new, new_buffer
                h, buffer_state = model.forward_step(h, x, buffer_state)
                
                # Apply Nudging
                if nudging and target_grads is not None:
                    h.sub_(beta * target_grads)

                # Convergence check
                diff = torch.norm(h - h_prev, dim=1).max()
                if diff < self.epsilon:
                    return h, {"steps": t+1, "converged": True, "buffer": buffer_state}

        return h, {"steps": self.max_steps, "converged": False, "buffer": buffer_state}
