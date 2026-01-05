import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models import LoopedTransformerBlock
from src.solver import EquilibriumSolver
from src.trainer import EqPropTrainer
import time
import argparse

def train(symmetric: bool = False):
    # Configuration - adjust beta for symmetric mode (needs smaller beta)
    config = {
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "batch_size": 128,
        "max_iters": 50,
        "damping": 0.9,
        "beta": 0.01 if symmetric else 0.1,  # Smaller beta for symmetric
        "lr": 1e-3,
        "epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "symmetric": symmetric,
        "attention_type": "linear" if symmetric else "linear"  # Linear for both
    }

    mode = "SYMMETRIC" if config["symmetric"] else "NON-SYMMETRIC"
    print(f"Training EqProp ({mode} mode) on {config['device']}")
    print(f"Beta: {config['beta']}, Attention: {config['attention_type']}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(784)) # Flatten 28x28
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model
    # MNIST sequence length: we treat the flattened image as a sequence?
    # Or just project it?
    # README says "MNIST (28x28 flattened to sequence)".
    # Usually this means we map 784 -> [seq, d_model].
    # But LoopedTransformerBlock expects [seq, batch, d_model] and processes it.
    # If we feed 784 tokens, it's very slow.
    # Maybe linear projection 784 -> d_model and seq_len=1?
    # "1-block looped transformer... Data: MNIST (28x28 flattened to sequence)"
    # If seq_len=1, it's just a ResNet/RNN.
    # Let's assume seq_len=1 for efficiency unless specified.
    # But Transformer implies sequence processing.
    # Maybe patches? 4x4 patches? 28x28 = 49 patches of 4x4.

    # Let's use a simple linear projection to d_model, and treat as seq_len=1 for now to start fast.
    # Or maybe seq_len=1 is implied by "h_t" vector?
    # "h_t (hidden)" usually means state.

    class MnistModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(784, config["d_model"])
            self.block = LoopedTransformerBlock(config["d_model"], config["n_heads"], config["d_ff"])

        def forward(self, h, x):
            # x is raw input [batch, 784].
            # We need to project x once?
            # Or is x part of the loop?
            # The solver calls model(h, x).
            # If we project x inside, we do it every step. Inefficient but okay.
            # Better: project x outside solver.
            # But solver signature is f(h, x).

            # Let's handle projection in the wrapper passed to solver?
            # Or inside model.

            # The input `x` to `LoopedTransformerBlock` expects [seq, batch, d_model].
            # If we project 784 -> [1, batch, d_model].
            pass

    # Actually, let's just project x outside.
    # And use seq_len=1.

    embedding = nn.Linear(784, config["d_model"]).to(config["device"])
    model = LoopedTransformerBlock(
        config["d_model"], config["n_heads"], config["d_ff"],
        attention_type=config["attention_type"],
        symmetric=config["symmetric"]
    ).to(config["device"])
    output_head = nn.Linear(config["d_model"], 10).to(config["device"])

    solver = EquilibriumSolver(
        max_iters=config["max_iters"],
        tol=1e-5,
        damping=config["damping"]
    )

    # Trainer needs the whole model.
    # But we have `embedding`, `model`, `output_head`.
    # EqPropTrainer expects `model` to be the recurrent part.
    # We can wrap `embedding` + `model`.

    class RecurrentWrapper(nn.Module):
        def __init__(self, embedding, block):
            super().__init__()
            self.embedding = embedding
            self.block = block

        def forward(self, h, x_raw):
            # x_raw: [batch, 784]
            # h: [seq, batch, d_model]

            # Project x if needed (only once? No, solver calls this iteratively).
            # If we pass projected x to solver, we don't need embedding here.

            # Let's project outside solver.
            return self.block(h, x_raw)

    # We will project x before passing to trainer.

    trainer = EqPropTrainer(model, solver, output_head, beta=config["beta"], lr=config["lr"])
    # Add embedding to optimizer
    trainer.optimizer.add_param_group({'params': embedding.parameters()})

    # Training Loop
    for epoch in range(config["epochs"]):
        model.train()
        embedding.train()
        output_head.train()

        total_loss = 0
        total_acc = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(config["device"]), target.to(config["device"])

            # Project data: [batch, 784] -> [batch, d_model] -> [1, batch, d_model]
            x_emb = embedding(data).unsqueeze(0)

            metrics = trainer.train_step(x_emb, target)

            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f} "
                      f"Iters: {metrics['iters_free']}/{metrics['iters_nudged']}")

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        duration = time.time() - start_time

        print(f"Epoch {epoch} Completed in {duration:.2f}s. Avg Loss: {avg_loss:.4f} Avg Acc: {avg_acc:.4f}")

        # Validation
        model.eval()
        embedding.eval()
        output_head.eval()
        test_acc = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config["device"]), target.to(config["device"])
                x_emb = embedding(data).unsqueeze(0)
                h0 = torch.zeros_like(x_emb)
                h_fixed, _ = solver.solve(model, h0, x_emb)
                y_pred = output_head(h_fixed.mean(dim=0))
                test_acc += (y_pred.argmax(-1) == target).float().mean().item()

        test_acc /= len(test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST with EqProp")
    parser.add_argument("--symmetric", action="store_true", 
                        help="Use symmetric mode (required for EqProp gradient equivalence)")
    args = parser.parse_args()
    train(symmetric=args.symmetric)
