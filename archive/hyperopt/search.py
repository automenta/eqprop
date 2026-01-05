import optuna
import torch
import torch.optim as optim
import time
import numpy as np
from src.models import LoopedMLP, ToroidalMLP, BackpropMLP, ModernEqProp, GatedMLP
from src.training import EqPropTrainer
from src.tasks import get_task_loader


def objective(trial, model_type="LoopedMLP", time_budget=None, epochs=1, 
              dataset_size=1000, task_name="mnist", device="cpu", seed=None):
    """Optuna objective function with improved hyperparameter ranges."""
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Hyperparameters - TIGHTENED RANGES based on experiments
    alpha = trial.suggest_float("alpha", 0.3, 0.8) if model_type != "BackpropMLP" else 0.5
    beta = trial.suggest_float("beta", 0.15, 0.25) if model_type != "BackpropMLP" else 0.0  # Optimal around 0.22
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    symmetric = trial.suggest_categorical("symmetric", [True, False]) if model_type == "LoopedMLP" else False
    buffer_decay = trial.suggest_float("buffer_decay", 0.7, 0.95) if model_type == "ToroidalMLP" else 0.9
    # CRITICAL: Spectral norm maintains L < 1 during training (verified experimentally)
    use_spectral_norm = trial.suggest_categorical("spectral_norm", [True]) if model_type != "BackpropMLP" else False
    
    # Dynamics parameters - TIGHTENED (equilibrium typically in <10 steps)
    if model_type != "BackpropMLP":
        max_steps = trial.suggest_categorical("max_steps", [10, 15, 20, 30])
        epsilon = trial.suggest_categorical("epsilon", [1e-3, 1e-4])
    else:
        max_steps = 30
        epsilon = 1e-4
    
    # Data
    train_loader, test_loader, input_dim, output_dim = get_task_loader(
        task_name, batch_size=64, dataset_size=dataset_size
    )
    
    # Model
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
    
    if model_type == "LoopedMLP":
        model = LoopedMLP(input_dim, hidden_dim, output_dim, alpha=alpha, 
                          symmetric=symmetric, use_spectral_norm=use_spectral_norm).to(device)
    elif model_type == "ToroidalMLP":
        model = ToroidalMLP(input_dim, hidden_dim, output_dim, alpha=alpha, 
                            decay=buffer_decay, use_spectral_norm=use_spectral_norm).to(device)
    elif model_type == "ModernEqProp":
        model = ModernEqProp(input_dim, hidden_dim, output_dim, gamma=alpha,
                            use_spectral_norm=use_spectral_norm).to(device)
    elif model_type == "GatedMLP":
        model = GatedMLP(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == "BackpropMLP":
        depth = trial.suggest_categorical("depth", [1, 2])
        model = BackpropMLP(input_dim, hidden_dim, output_dim, depth=depth).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Training Loop
    trainer = EqPropTrainer(model, optimizer, beta=beta, alpha=alpha, 
                            epsilon=epsilon, max_steps=max_steps)
    
    start_time = time.time()
    total_converged = 0
    total_steps = 0
    
    for epoch in range(epochs): 
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            if model_type == "BackpropMLP":
                optimizer.zero_grad()
                out = model(x)
                loss = torch.nn.functional.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
            else:
                metrics = trainer.step(x, y)
                # Track convergence
                if 'converged_free' in metrics:
                    total_converged += int(metrics.get('converged_free', False))
                total_steps += 1
                
            # Check time budget
            if time_budget and (time.time() - start_time) > time_budget:
                break
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
        accuracy = correct / total
            
    param_count = sum(p.numel() for p in model.parameters())
    elapsed = time.time() - start_time
    convergence_rate = total_converged / max(total_steps, 1) if model_type != "BackpropMLP" else 1.0

    return accuracy, elapsed, param_count, convergence_rate


def run_study(study_name, model_type="LoopedMLP", n_trials=10, time_budget=60, 
              epochs=1, dataset_size=1000, task_name="mnist", n_seeds=1):
    """Run hyperparameter study with optional multi-seed aggregation."""
    
    full_study_name = f"{task_name}_{study_name}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Multi-objective: Maximize Acc, Minimize Time, Minimize Params, Maximize Convergence
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize", "maximize"], 
        study_name=full_study_name, 
        storage=f"sqlite:///{full_study_name}.db", 
        load_if_exists=True
    )
    
    def func(trial):
        if n_seeds > 1:
            # Multi-seed: aggregate results
            accs, times, params_list, conv_rates = [], [], [], []
            for seed in range(n_seeds):
                acc, dur, params, conv = objective(
                    trial, model_type, time_budget, epochs=epochs, 
                    dataset_size=dataset_size, task_name=task_name, 
                    device=device, seed=seed
                )
                accs.append(acc)
                times.append(dur)
                params_list.append(params)
                conv_rates.append(conv)
            
            # Return mean values
            return np.mean(accs), np.mean(times), params_list[0], np.mean(conv_rates)
        else:
            return objective(
                trial, model_type, time_budget, epochs=epochs, 
                dataset_size=dataset_size, task_name=task_name, device=device
            )
        
    study.optimize(func, n_trials=n_trials, timeout=time_budget * 5) 
    
    print(f"\nPareto Front for {model_type}:")
    for t in study.best_trials:
        acc, dur, params, conv = t.values
        print(f"  Trial {t.number}: Acc={acc:.4f}, Time={dur:.2f}s, Params={params}, Conv={conv:.2f}, Config={t.params}")
        
    # Return best accuracy among pareto optimal trials
    best_trial = max(study.best_trials, key=lambda t: t.values[0])
    best_acc = best_trial.values[0]
    best_time = best_trial.values[1]
    best_params = int(best_trial.values[2])
    
    return best_acc, best_time, best_params
