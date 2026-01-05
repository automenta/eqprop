import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
import numpy as np

def test_gradient_equivalence():
    torch.manual_seed(42)

    # Configuration
    d_model = 64
    n_heads = 4
    d_ff = 256
    seq_len = 10
    batch_size = 5
    beta = 0.01  # Try smaller beta for better approximation? 0.01 is okay.

    # Setup
    model = LoopedTransformerBlock(d_model, n_heads, d_ff, attention_type='linear')
    # Output head: maps [d_model] to [n_classes]
    n_classes = 10
    output_head = nn.Linear(d_model, n_classes)

    # Solver
    solver = EquilibriumSolver(max_iters=50, tol=1e-6, damping=0.9)

    # Inputs
    x = torch.randn(seq_len, batch_size, d_model)
    y = torch.tensor([0, 1, 2, 3, 4])

    # --- BP Gradient Calculation ---
    model.zero_grad()
    output_head.zero_grad()

    # BPTT through the solver for baseline
    h0 = torch.zeros_like(x)
    h_bp, _ = solver.solve(model, h0, x)

    y_pred_bp = output_head(h_bp.mean(dim=0))
    loss_bp = F.cross_entropy(y_pred_bp, y)
    loss_bp.backward()

    grads_bp = {}
    for name, param in model.named_parameters():
        grads_bp[name] = param.grad.clone()
    for name, param in output_head.named_parameters():
        if param.grad is not None:
            grads_bp['head.'+name] = param.grad.clone()

    # --- EqProp Gradient Calculation ---
    model.zero_grad()
    output_head.zero_grad()

    # 1. Free phase
    # We need h_free value.
    with torch.no_grad():
        h_free, _ = solver.solve(model, h0, x)

    # 2. Nudged phase
    def nudged_dynamics(h, x):
        h = h.detach().requires_grad_(True)
        h_new = model(h, x)
        y_pred = output_head(h_new.mean(dim=0))
        loss = F.cross_entropy(y_pred, y)
        # Nudge: + beta * grad(-Loss) = - beta * grad(Loss)
        grads = torch.autograd.grad(-loss, h_new, create_graph=True, retain_graph=True)[0]
        return h_new + beta * grads

    h_nudged, _ = solver.solve(nudged_dynamics, h_free.detach(), x)

    # 3. Update (Corrected "Implicit Differentiation" via EqProp)

    # For Transformer parameters:
    # Gradient approx = (h_free - h_nudged) / beta * df/dtheta
    # This assumes h_nudged = h_free + beta * (I-J)^-1 * (-dL/dh)
    # So h_free - h_nudged = beta * (I-J)^-1 * dL/dh
    # Wait, dL/dtheta = dL/dh * (I-J)^-1 * df/dtheta
    # So (h_free - h_nudged)/beta is roughly dL/dh * (I-J)^-1 ?
    # Actually, Scellier proof says (h_beta - h_0)/beta converges to the "adjoint" state.
    # The adjoint state lambda satisfies lambda = J^T lambda - dL/dh.
    # Actually, lambda = (I - J^T)^-1 (-dL/dh).
    # And dL/dtheta = lambda^T df/dtheta.
    # So we need lambda.
    # h_beta - h_0 approx beta * (I-J)^-1 * (-dL/dh).
    # Note that (I-J)^-1 is generally NOT equal to (I-J^T)^-1 unless J is symmetric.
    # For symmetric weights (Hopfield/Energy models), J is symmetric!
    # BUT Transformer is NOT symmetric (Attention is not symmetric, weights are not tied W_xy = W_yx).

    # CRITICAL REALIZATION:
    # EqProp strictly works for Energy Based Models (Symmetric weights).
    # The README claims "Toroidal Equilibrium Propagation for Transformers".
    # And "Hypothesis H1: ... with gradients equivalent to implicit differentiation".
    # AND "Testable Prediction 1: EqProp gradient = BP gradient".

    # If the transformer is not symmetric, EqProp (Scellier 2017) gradients will NOT match BP gradients generally.
    # Unless we enforce symmetry or use a variant (Vector Field EqProp / Generalized EqProp).
    # Generalized EqProp (Laborieux et al) requires a distinct "backward" relaxation phase with separate dynamics.
    # The README does NOT describe a separate backward phase dynamics (e.g. using W_transpose).
    # It describes "Nudged Phase" using the SAME dynamics + nudge.

    # If using same dynamics, we are assuming symmetric J?
    # Or maybe the "Contrastive Hebbian" update proposed is just an approximation?
    # BUT "Success Criteria: Cosine sim > 0.99".
    # This implies exact matching is expected.

    # If exact matching is expected with Algorithm 1, then either:
    # A) The architecture is symmetric (Weight tied? "weight-tied (toroidal)").
    #    "Weight-tied" usually means weights are shared across *time* (layers).
    #    It does NOT mean W_ij = W_ji within a layer.
    # B) The implementation of EqProp in the README handles the asymmetry?
    #    Algorithm 1: "Nudged Phase... h' <- (1-alpha)h + alpha*f(h,x) + beta*grad(L)".
    #    This uses f(h,x) again. This is "Recirculation" or "Target Prop" essentially?
    #    If we use Recirculation (backprop through the reconstruction), it approximates gradients.

    # Let's try the "Target Prop" update I derived:
    # delta = (h_free - h_nudged) / beta.
    # backward(model(h_free), delta).
    # This computes: delta^T * df/dtheta.
    # = (1/beta) (h_free - h_nudged)^T * df/dtheta.
    # approx (I-J)^-1 dL/dh * df/dtheta.
    # We want dL/dh * (I-J)^-1 * df/dtheta.
    # These are only equal if (I-J)^-1 commutes with everything or J is symmetric.

    # So, for non-symmetric networks, standard EqProp DOES NOT WORK.
    # However, maybe the "Looped Transformer" is somehow special?
    # Or maybe I should just implement what's requested and report the failure?
    # But the prompt is "Implement...".

    # Wait, there's "Algorithm 1b: Purely Local Hebbian Nudging".
    # And the main Algo 1 says "Weight Update: Contrastive Hebbian".
    # Maybe the key is `A_beta * A_beta` terms.
    # `(A^beta_l * A^beta_l - A*_l * A*_l)`.
    # This looks like Energy based update.
    # If the network is NOT Energy based, this update is heuristic.

    # But wait, Scellier 2019 "Equilibrium Propagation for Arbitrary Recurrent Neural Networks"?
    # Requires "Vector Field EqProp" involving a second "backward" phase with transposed weights.
    # The README does not mention transposed weights or a backward phase with different dynamics.

    # Let's check "Architecture -> Attention Variants".
    # "Softmax attention may violate contraction... Include linear attention...".
    # "Energy Formulation for Attention... Open Problem".
    # This implies the author knows standard transformers are not Energy models.
    # "Appendix A1... Candidates: Linear attention surrogate ... admits energy".

    # If I use Softmax attention (default in models.py), it has no Energy.
    # So EqProp guarantees don't hold?
    # Maybe that's why cosine sim is bad.

    # I should try `Linear Attention`?
    # The README implementation says: "Softmax attention ... Primary (if stable)".
    # "Gradient Verification ... Success: Cosine sim > 0.99".

    # Maybe I should interpret the "Simplified" proxy loss differently.
    # `loss_proxy = ((h_nudged - h_free) ** 2)`.
    # If we treat `h_nudged` as a "target" for `h_free`, and we want to move `h_free` towards it.
    # But `h_free` is determined by `theta`.
    # So we minimize `|| h_free(theta) - h_nudged.detach() ||^2`.
    # Gradient: `2(h_free - h_nudged) * dh_free/dtheta`.
    # This is `2 * beta * (h_free - h_nudged)/beta * dh_free/dtheta`.
    # If `(h_free - h_nudged)/beta` is the error signal `delta`.
    # Then this is `2 * beta * delta * dh_free/dtheta`.
    # This matches the "Target Prop" update I derived, except for the `dh_free/dtheta` term.
    # `dh_free/dtheta` is the sensitivity of the fixed point.
    # `dh_free/dtheta = (I-J)^{-1} df/dtheta`.
    # So this update computes: `delta^T (I-J)^{-1} df/dtheta`.
    # Since `delta ~ (I-J)^{-1} dL/dh`.
    # Update ~ `dL/dh (I-J)^{-1} (I-J)^{-1} df/dtheta`.
    # This has an extra `(I-J)^{-1}` !!

    # Unless... `delta` is NOT `(I-J)^{-1} dL/dh`?

    # Let's go back to the code.
    # I will try the "Target Prop" approach:
    # `loss_surrogate = ((model(h_free.detach(), x) - h_nudged.detach()) ** 2).mean()`
    # No, that's just training one step to predict h_nudged.
    # `delta = (h_free_detached - h_nudged.detach()) / beta`.
    # `backward(model(h_free_detached), delta)`.

    # Let's try this in the script.

    h_free_detached = h_free.detach()
    h_out = model(h_free_detached, x)

    # Delta: (h_free - h_nudged) / beta?
    # We want to minimize Loss.
    # h_nudged is "better".
    # We want h_out (which approximates h_free) to move towards h_nudged.
    # Vector: h_nudged - h_free.
    # We want -Gradient of Loss.
    # So we want to maximize correlation with (h_nudged - h_free)?
    # Or minimize distance?
    # Minimizing distance `|| h_out - h_nudged ||^2` gives gradient `2(h_out - h_nudged)`.
    # This pushes h_out towards h_nudged.
    # Correct.
    # So we can just use MSE loss between h_out and h_nudged.
    # And scale by 1/beta?
    # If `h_nudged - h_free ~ beta * grad`.
    # MSE grad ~ `beta * grad`.
    # We want `grad`. So divide by beta.
    # Also divide by 2?

    # We will try:
    # loss_update = 1/beta * MSE(h_out, h_nudged.detach())
    # loss_update.backward()

    loss_update = (1.0 / beta) * F.mse_loss(h_out, h_nudged.detach())
    loss_update.backward()

    # For head weights, we use standard loss at h_free.
    y_pred_free = output_head(h_free_detached.mean(dim=0))
    loss_head = F.cross_entropy(y_pred_free, y)
    loss_head.backward()

    grads_eq = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_eq[name] = param.grad.clone()
    for name, param in output_head.named_parameters():
        if param.grad is not None:
            grads_eq['head.'+name] = param.grad.clone()

    # Comparison
    print(f"Beta: {beta}")
    print("-" * 20)

    similarities = []
    keys = set(grads_bp.keys()) & set(grads_eq.keys())

    for name in sorted(keys):
        g_bp = grads_bp[name]
        g_eq = grads_eq[name]

        if g_bp.norm() == 0 or g_eq.norm() == 0:
            print(f"{name}: Zero norm")
            continue

        cos_sim = F.cosine_similarity(g_bp.flatten().unsqueeze(0), g_eq.flatten().unsqueeze(0)).item()
        neg_cos_sim = F.cosine_similarity(g_bp.flatten().unsqueeze(0), -g_eq.flatten().unsqueeze(0)).item()

        similarities.append(cos_sim)

        print(f"{name}: Cosine Sim = {cos_sim:.4f} | Negated Sim = {neg_cos_sim:.4f}")

    avg_sim = np.mean(similarities)
    print(f"Average Cosine Similarity: {avg_sim:.4f}")

    return avg_sim

def test_gradient_equivalence_symmetric():
    """Test gradient equivalence with symmetric mode (required for EqProp theory).
    
    Symmetric mode implements weight tying:
    - W_out = W_q^T (attention output projection)
    - W_k = W_v (key and value share weights)
    - W2 = W1^T (FFN output = input weight transposed)
    
    These constraints ensure the network has symmetric Jacobian, which is required
    for Scellier & Bengio 2017's EqProp gradient equivalence theorem to hold.
    
    Reference: Scellier & Bengio (2017), "Equilibrium Propagation: Bridging the
    Gap between Energy-Based Models and Backpropagation"
    """
    torch.manual_seed(42)

    # Configuration
    d_model = 64
    n_heads = 4
    d_ff = 256
    seq_len = 10
    batch_size = 5
    beta = 0.001  # Small beta for better gradient approximation

    # Setup with symmetric mode
    model = LoopedTransformerBlock(d_model, n_heads, d_ff, 
                                    attention_type='linear', symmetric=True)
    n_classes = 10
    output_head = nn.Linear(d_model, n_classes)

    # Solver
    solver = EquilibriumSolver(max_iters=50, tol=1e-6, damping=0.9)

    # Inputs
    x = torch.randn(seq_len, batch_size, d_model)
    y = torch.tensor([0, 1, 2, 3, 4])

    # --- BP Gradient Calculation ---
    model.zero_grad()
    output_head.zero_grad()

    h0 = torch.zeros_like(x)
    h_bp, _ = solver.solve(model, h0, x)

    y_pred_bp = output_head(h_bp.mean(dim=0))
    loss_bp = F.cross_entropy(y_pred_bp, y)
    loss_bp.backward()

    grads_bp = {}
    for name, param in model.named_parameters():
        grads_bp[name] = param.grad.clone()
    for name, param in output_head.named_parameters():
        if param.grad is not None:
            grads_bp['head.'+name] = param.grad.clone()

    # --- EqProp Gradient Calculation ---
    model.zero_grad()
    output_head.zero_grad()

    # Free phase
    with torch.no_grad():
        h_free, _ = solver.solve(model, h0, x)

    # Nudged phase
    def nudged_dynamics(h, x):
        h = h.detach().requires_grad_(True)
        h_new = model(h, x)
        y_pred = output_head(h_new.mean(dim=0))
        loss = F.cross_entropy(y_pred, y)
        # Nudge in direction that decreases loss
        grads = torch.autograd.grad(loss, h_new, create_graph=True, retain_graph=True)[0]
        return h_new - beta * grads

    h_nudged, _ = solver.solve(nudged_dynamics, h_free.detach(), x)

    # MSE proxy update
    h_free_detached = h_free.detach()
    h_out = model(h_free_detached, x)
    
    loss_update = (1.0 / beta) * F.mse_loss(h_out, h_nudged.detach())
    loss_update.backward()

    # Output head gradient
    y_pred_free = output_head(h_free_detached.mean(dim=0))
    loss_head = F.cross_entropy(y_pred_free, y)
    loss_head.backward()

    grads_eq = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads_eq[name] = param.grad.clone()
    for name, param in output_head.named_parameters():
        if param.grad is not None:
            grads_eq['head.'+name] = param.grad.clone()

    # Comparison
    print(f"Beta: {beta}")
    print(f"Symmetric Mode: True")
    print("-" * 20)

    similarities = []
    keys = set(grads_bp.keys()) & set(grads_eq.keys())

    for name in sorted(keys):
        g_bp = grads_bp[name]
        g_eq = grads_eq[name]

        if g_bp.norm() == 0 or g_eq.norm() == 0:
            print(f"{name}: Zero norm")
            continue

        cos_sim = F.cosine_similarity(g_bp.flatten().unsqueeze(0), g_eq.flatten().unsqueeze(0)).item()
        similarities.append(cos_sim)
        print(f"{name}: Cosine Sim = {cos_sim:.4f}")

    avg_sim = np.mean(similarities)
    print(f"Average Cosine Similarity: {avg_sim:.4f}")

    return avg_sim

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Non-Symmetric Mode")
    print("=" * 60)
    avg_sim_nonsym = test_gradient_equivalence()
    
    print("\n" + "=" * 60)
    print("Testing Symmetric Mode (EqProp Theoretical Requirement)")
    print("=" * 60)
    avg_sim_sym = test_gradient_equivalence_symmetric()
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Non-symmetric avg cosine similarity: {avg_sim_nonsym:.4f}")
    print(f"Symmetric avg cosine similarity:     {avg_sim_sym:.4f}")
    print()
    
    # Success criteria from TODO.md
    if avg_sim_sym > 0.99:
        print("✓ Symmetric mode PASSED (cosine sim > 0.99)")
    else:
        print("✗ Symmetric mode FAILED (cosine sim <= 0.99)")
